import numpy as np
import pandas as pd

from main import Main

choice_dict = {0 : 'single', 1 : 'double'}

choice = np.random.choice(2)
story, question, correct_answer, weights1, weights2, predicted_answer = Main.generateAnswer(choice_dict[choice])
