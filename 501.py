
def update_score(current_score, dart_scores):
    new_score = current_score
    for score in dart_scores:
        if score is not None:
            new_score -= score
    
    if new_score < 0:
        print("Bust!")
        return current_score  # Return the original score (before the bust)
    else:
        return new_score

def check_game_over(current_score):
    return current_score == 0

def play_game_501(gui):
    while not check_game_over(gui.current_score):
        # Wait for the dart scores to be updated
        while None in gui.dart_scores:
            pass

        # Update the current score based on the dart scores
        gui.current_score = update_score(gui.current_score, gui.dart_scores)

        # Reset the dart scores
        gui.dart_scores = [None] * 3

        # Check if the game is over
        if check_game_over(gui.current_score):
            print("Game Over! You won!")
            break
