import numpy as np
import random

class SocialProofGenerator:
    def __init__(self):
        # Configuration based on typical E-Commerce datasets (like Amazon)
        # Ratings are J-shaped: heavily skewed towards 4.5-5.0
        self.rating_probs = [0.05, 0.05, 0.15, 0.35, 0.40] # Prob for 1, 2, 3, 4, 5 stars
        self.rating_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
    def generate(self):
        """
        Generates a Rating/Review pair that is statistically realistic 
        but causally independent of the product's actual quality.
        """
        # 1. Generate Base Rating (J-Shaped Distribution)
        # Real ratings are rarely integers, so we add jitter
        base_star = np.random.choice(self.rating_values, p=self.rating_probs)
        # Add jitter (e.g. 4.0 -> 4.3) but clip to 5.0
        jitter = np.random.uniform(0, 0.9)
        final_rating = min(round(base_star + jitter, 1), 5.0)
        
        # 2. Generate Review Count (Power Law / Pareto Distribution)
        # Most items have < 50 reviews. A few have 10,000+.
        # Pareto shape parameter alpha=1.16 is common for web activity
        review_count = int(np.random.pareto(a=1.5) * 50) + 1
        
        # Cap at realistic max (e.g., 50k) to prevent LLM token weirdness
        review_count = min(review_count, 50000)
        
        return final_rating, review_count

# Usage in run_simulation.py:
# generator = SocialProofGenerator()
# rating, count = generator.generate()
# social_proof = f"Rating: {rating}/5 ({count} reviews)"