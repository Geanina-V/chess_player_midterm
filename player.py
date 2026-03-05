import chess
import random
import re
import torch
from typing import Optional, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from chess_tournament.players import Player




class TransformerPlayer(Player):
    """
    A Distilgpt2 transfomer-based model chess player.
    This model uses the following techniques to make its decisions:

    1. Heuristics: apply chess principles when choosing which one is the best move by scoring the moves

    2. Enhanced Prompting: few Fen examples to guide the decoder model

    3. Generated text attempts: Try several times to give a good output by increasing temperature

    4. Smart opening: when opening a game make sure the model uses strong valid openings
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    # Common opening moves to prefer (e4, d4, Nf3, c4) and less common but used (b3, g3, f4) -->source: https://en.wikipedia.org/wiki/Chess_opening
    WHITE_OPENING_MOVES = ['e2e4', 'd2d4', 'g1f3', 'c2c4', 'e2e3', 'd2d3', 'b2b3', 'g2g3', 'f2f4']
    BLACK_OPENING_MOVES = ['e7e5', 'd7d6', 'g8f6', 'c7c5', 'e7e6', 'd7d5', 'b7b6', 'g7g6', 'f7f5']

    def __init__(
        self,
        name: str = "TransformerPlayer",
        model_id: str = "distilbert/distilgpt2",
        temperature: float = 0.3,
        max_new_tokens: int = 12,
        use_smart_opening: bool = True,
        n_attempts: int = 3,

    ):
        super().__init__(name)

        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.use_smart_opening = use_smart_opening
        self.n_attempts = n_attempts

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        self.tokenizer = None
        self.model = None
        self.scorer = None

    def _load_model(self):
        if self.model is None:
            print(f"[{self.name}] Loading {self.model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

    def _build_enhanced_prompt(self, fen: str) -> str:
        """
        Enhanced prompt with examples of input.

        Shows the model what we expect: FEN position --> UCI move
        """
        return f"""Imagine you are an expert chess player. 
        Please suggest best next move in UCI format, by following the below examples:

FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Move: e2e4

FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
Move: e7e5

FEN: rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2
Move: g1f3

FEN: rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2
Move: b8c6

FEN: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
Move: f1c4

FEN: r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3
Move: f8c5

FEN: {fen}
Move:"""

    def _get_legal_moves(self, fen: str) -> List[str]:
        """Get list of legal moves in UCI format."""
        try:
            board = chess.Board(fen)
            return [move.uci() for move in board.legal_moves]
        except:
            return []

    def _is_opening_position(self, fen: str) -> bool:
        """Checking if we're still in the opening."""
        try:
            board = chess.Board(fen)
            return board.fullmove_number == 1
        except:
            return False

    def _score_move_heuristic(self, board: chess.Board, move: chess.Move) -> float:
        """
        Using best chess principles to build heuristics for making smart decision
        on which move is best to make next
        """
        score = 0.0
        
        #copying the board and making move to evaluate
        board_copy = board.copy()
        board_copy.push(move)
        
        #checking if we are checking the opponent's king
        if board_copy.is_check():
            score += 30.0
        
        # checking if we are capturing opponent's piece and give higher value to more important pieces
        if board.is_capture(move):
            piece_values = {
                chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
            }
            captured = board.piece_at(move.to_square) #checking captured piece
            if captured:
                captured_value = piece_values.get(captured.piece_type, 0)
                score += captured_value * 12 #multiply the value of the piece to the score
        
        #making sure that we do not lose more valuable pieces
        moving_piece = board.piece_at(move.from_square)
        if moving_piece and board_copy.is_attacked_by(not board.turn, move.to_square):
            piece_values = {
                chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
            }
            piece_value = piece_values.get(moving_piece.piece_type, 0)
            
            # Checking if valuable piece is defended
            if not board_copy.is_attacked_by(board.turn, move.to_square):
                #if piece is not defended, then multiply value by score for higher penalty
                score -= piece_value * 15
        
        # We want to keep control of the center of the board --> e4, d4, e5, d5
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        if move.to_square in center_squares:
            score += 6.0
        
        
        #check if there is piece development in first 10 moves
        if board.fullmove_number <= 10:
            if moving_piece and moving_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                from_rank = chess.square_rank(move.from_square) #get the row of the square where the piece is
                
                #checking the piece that is moving, it is moving from its original back row.
                if (board.turn == chess.WHITE and from_rank == 0) or \
                   (board.turn == chess.BLACK and from_rank == 7):
                    score += 8.0
            
            #give penalty if the same piece is moved twice
            if len(board.move_stack) > 0:
                last_move = board.peek()
                if move.from_square == last_move.to_square:
                    score -= 5.0
        
        #check if pawns are advancing in endgame and give higher score to encorage promotion to Queen piece
        if board.fullmove_number > 30 and moving_piece and \
           moving_piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(move.to_square)
            if (board.turn == chess.WHITE and to_rank >= 5) or \
               (board.turn == chess.BLACK and to_rank <= 2):
                score += 5.0
        
        return score

    def _get_move_by_generation(self, fen: str, legal_moves: List[str]) -> Optional[str]:
        """
        Generate move using text generation with multiple attempts and different temperature at each attempt.
        """
        prompt = self._build_enhanced_prompt(fen)

        # Try multiple times with varying parameters
        for attempt in range(self.n_attempts):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # Gradually increase temperature for more diversity
                current_temp = self.temperature + (attempt * 0.2)


                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=min(current_temp, 1),  # Cap at 1 to avoid too much randomness
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                #if the model outputs a string with more than UCI, extract just the UCI
                if decoded.startswith(prompt):
                    decoded = decoded[len(prompt):].strip()

                # Try to extract move
                match = self.UCI_REGEX.search(decoded)
                if match:
                    move = match.group(1).lower()
                    if move in legal_moves:
                        return move

                # Check if any legal move appears anywhere in output
                for legal_move in legal_moves:
                    if legal_move in decoded.lower():
                        return legal_move
            
            except Exception:
              continue

        return None
    
    def _get_move_by_heuristics_and_generation(self, fen: str, legal_moves: List[str]) -> Optional[str]:
        """
        Combine heuristics with text generation:
        1. First we get the best generated text
        2. If it produces a move, validate it with heuristics
        3. If move is terrible (negative heuristic score), try heuristic-based selection instead only
        4. Otherwise use the generated move
        """
        board = chess.Board(fen)
        
        # Try text generation first
        generated_move = self._get_move_by_generation(fen, legal_moves)
        
        if generated_move:
            # Check if the generated move is bad
            move_obj = chess.Move.from_uci(generated_move)
            heuristic_score = self._score_move_heuristic(board, move_obj)
            
            # If heuristic score is very negative reject it
            if heuristic_score < -20:
                # use heuristic selection instead
                return self._get_best_heuristic_move(fen, legal_moves)
            else:
                # Generated move is acceptable
                return generated_move
        
        # Text generation failed, use heuristics
        return self._get_best_heuristic_move(fen, legal_moves)

    def _get_best_heuristic_move(self, fen: str, legal_moves: List[str]) -> Optional[str]:
        """
        Select best move using heuristics only.
        Pick from top 3 to add variety.
        """
        if not legal_moves:
            return None
        
        try:
            board = chess.Board(fen)
            scored = [
                (move, self._score_move_heuristic(board, chess.Move.from_uci(move)))
                for move in legal_moves
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Pick randomly from top 3
            top = scored[:min(3, len(scored))]
            return random.choice(top)[0]
        except:
            return random.choice(legal_moves)

    def _smart_random_move(self, fen: str, legal_moves: List[str]) -> Optional[str]:
        """
        prefer common opening moves if in opening.
        Otherwise, random legal move.
        """
        if not legal_moves:
            return None

        try:
          board = chess.Board(fen)
        except:
          return random.choice(legal_moves)

        # If in opening, prefer standard opening moves
        if self.use_smart_opening and self._is_opening_position(fen):
            # Find which opening moves are legal
            if board.turn == chess.WHITE:
              available_opening_moves = [m for m in self.WHITE_OPENING_MOVES if m in legal_moves]
              if available_opening_moves:
                return random.choice(available_opening_moves)
            else:
              available_opening_moves = [m for m in self.BLACK_OPENING_MOVES if m in legal_moves]
              if available_opening_moves:
                return random.choice(available_opening_moves)

        # Otherwise, random legal move
        return random.choice(legal_moves)

    def get_move(self, fen: str) -> Optional[str]:

        try:
            self._load_model()
        except Exception as e:
            # Can't load model --> just return random legal move
            legal_moves = self._get_legal_moves(fen)
            return random.choice(legal_moves) if legal_moves else None

        # Get legal moves
        legal_moves = self._get_legal_moves(fen)
        if not legal_moves:
            return None

        # Strategy 1: if in opening prefer strong opening moves
        if self.use_smart_opening and self._is_opening_position(fen):
          opening_move = self._smart_random_move(fen, legal_moves)
          if opening_move:
            return opening_move

        # Strategy 2: use heuristics + text generation
        move = self._get_move_by_heuristics_and_generation(fen, legal_moves)
        if move:
            return move

        #Strategy 3: fallback on heuristic move if nothing else works
        return self._get_best_heuristic_move(fen, legal_moves)
        
