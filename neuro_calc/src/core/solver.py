from collections import deque
from typing import List, Optional, Tuple
import math

class GestureSolver:
    """
    State Machine for converting continuous gesture streams into discrete mathematical operations.
    Implements a robust 'Debouncing' mechanism to filter model noise.
    """
    def __init__(self, stability_frames: int = 15, confidence_threshold: float = 0.85):
        # Configuration
        self.stability_frames = stability_frames
        self.conf_thresh = confidence_threshold
        
        # Mapping (Must match your Training Labels!)
        self.vocab = {
            0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 
            5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
            10: "+", 11: "-", 12: "*", 13: "/", 14: "="
        }
        
        # State Management
        self.history_buffer = deque(maxlen=stability_frames)
        self.command_stack: List[str] = [] # e.g. ["3", "+", "5"]
        self.last_committed_token: Optional[str] = None
        self.current_result: Optional[str] = None
        
        # Cooldown prevents rapid-fire triggering of the same number
        self.cooldown_counter = 0

    def process_frame(self, class_idx: int, probability: float) -> Tuple[str, str]:
        """
        Ingest a single frame prediction.
        Returns: (Current Equation String, Result String)
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self._get_display_strings()

        # 1. Filter Low Confidence
        if probability < self.conf_thresh:
            self.history_buffer.clear() # Reset stability if uncertain
            return self._get_display_strings()

        # 2. Add to History
        token = self.vocab.get(class_idx, "?")
        self.history_buffer.append(token)

        # 3. Check Stability (Are all frames in buffer identical?)
        if len(self.history_buffer) == self.stability_frames:
            if len(set(self.history_buffer)) == 1:
                stable_token = self.history_buffer[0]
                self._commit_token(stable_token)

        return self._get_display_strings()

    def _commit_token(self, token: str):
        """
        Logic to accept a token into the equation stack.
        """
        # Hysteresis: Don't repeat the same token immediately (unless it's a number after an op)
        if token == self.last_committed_token:
            return

        self.last_committed_token = token
        
        # Logic Flow
        is_digit = token.isdigit()
        is_op = token in ["+", "-", "*", "/"]
        is_eq = token == "="

        if is_digit:
            # If we just finished an equation, start new
            if self.current_result is not None:
                self.command_stack = [token]
                self.current_result = None
            else:
                # Basic State: Expecting Number
                # If stack is empty or last was Op, add Number
                if not self.command_stack or self.command_stack[-1] in ["+", "-", "*", "/"]:
                    self.command_stack.append(token)
                
        elif is_op:
            # Can only add Op if we have a LHS number
            if self.command_stack and self.command_stack[-1].isdigit():
                self.command_stack.append(token)
                
        elif is_eq:
            # Trigger Calculation
            if len(self.command_stack) >= 3:
                self._evaluate()

        # Reset buffer to prevent double triggering
        self.history_buffer.clear()
        self.cooldown_counter = 20 # Wait 20 frames before accepting new input

    def _evaluate(self):
        """
        Solves the stack: [Num, Op, Num]
        """
        try:
            # Safe evaluation string construction
            expr = "".join(self.command_stack)
            
            # Security: Ensure only valid chars are processed
            if not all(c.isdigit() or c in "+-*/" for c in expr):
                return

            res = eval(expr) # Safe here due to strict vocab control
            
            # Format output
            if isinstance(res, float):
                self.current_result = f"{res:.2f}"
            else:
                self.current_result = str(res)
                
            # Clear stack for next op
            self.command_stack = [] 
            
        except ZeroDivisionError:
            self.current_result = "ERR: Div0"
            self.command_stack = []

    def _get_display_strings(self) -> Tuple[str, str]:
        equation = " ".join(self.command_stack)
        result = self.current_result if self.current_result else ""
        return equation, result