import chess
import random
import re
import torch
from typing import Optional, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from minicons import scorer
from chess_tournament.players import Player




class TransformerPlayer(Player):
    """
    A Distilgpt2 transfomer-based model chess player.
    This model uses the following techniques to make its decisions:

    1. Probability Scoring (minicons): score all legal moves by model probability and 20% chance to pick one of the top 5 best moves

    2. Enhanced Prompting: few Fen examples to guide the decoder model

    3. Multiple Sampling: Try several times to make the right decision by increasing temperature --> randomness

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
        temperature: float = 0.2,
        max_new_tokens: int = 12,
        use_probability_scoring: bool = True,
        use_smart_opening: bool = True,
        n_attempts: int = 5,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.use_probability_scoring = use_probability_scoring
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

            
            if self.use_probability_scoring:
                try:
                    self.scorer = scorer.IncrementalLMScorer(model=self.model,tokenizer=self.tokenizer)
                except Exception as e:
                    self.scorer = None

    def _build_enhanced_prompt(self, fen: str) -> str:
        """
        Enhanced prompt with examples of input.

        Shows the model what we expect: FEN position --> UCI move
        """
        return f"""Chess: predict next move in UCI format.

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

    def _score_moves_by_probability(self, fen: str, legal_moves: List[str]) -> List[Tuple[str, float]]:
        """
        Assigning probabilities scores to moves using minicons.
        This way we can make the best move based on the model's understanding and not just a random one.
        """
        if self.scorer is None:
            return []

        try:
            prompt = self._build_enhanced_prompt(fen)

            # Create full sequences: prompt + move
            sequences = [f"{prompt} {move}" for move in legal_moves]

            # Scoring each sequence
            scores = self.scorer.sequence_score(sequences, reduction=lambda x: -x.sum(0).item())

            # sorting scores from highest to lowest
            move_scores = list(zip(legal_moves, scores))
            move_scores.sort(key=lambda x: x[1], reverse=True)

            return move_scores

        except Exception as e:
            return []

    def _get_move_by_scoring_with_exploration(self, fen: str, legal_moves: List[str]) -> Optional[str]:
      """We get the best move based on the scores.
      To avoid the model being deterministic and allowing other moves,
      20% of the times we use other moves from top 5 moves by using random choice."""
      scored_moves = self._score_moves_by_probability(fen, legal_moves)

      if not scored_moves:
          return None

      # Add exploration (20% chance to try other good moves)
      if random.random() < 0.2:  # 20% exploration rate
          # Sample from top 5 moves
          top_5 = scored_moves[:min(5, len(scored_moves))]
          moves = [move for move, _ in top_5]
          chosen_move = random.choices(moves)[0] #choose one random move of top 5
          return chosen_move

      # Otherwise, return best move
      return scored_moves[0][0]

    def _get_move_by_generation(self, fen: str, legal_moves: List[str]) -> Optional[str]:
        """
        Generate move using text generation with multiple attempts.
        """
        prompt = self._build_enhanced_prompt(fen)

        # Try multiple times with varying parameters
        for attempt in range(self.n_attempts):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # Gradually increase temperature for more diversity --> higher chances of getting a valid output
                current_temp = self.temperature + (attempt * 0.15)

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
            # Can't load model --> just return smart random
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

        # Strategy 2: Probability scoring and 20% chance random pick of top 5 moves
        if self.use_probability_scoring and self.scorer is not None:
            move = self._get_move_by_scoring_with_exploration(fen, legal_moves)
            if move:
                return move

        # Strategy 3: Text generation with smart prompting
        move = self._get_move_by_generation(fen, legal_moves)
        if move:
            return move

        # Strategy 4: random legal move (if all fails above)
        return random.choice(legal_moves) if legal_moves else None

