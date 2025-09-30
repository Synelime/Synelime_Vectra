

This project implements a personalized tutoring system that recommends study paths to students based on their performance. The system uses reinforcement learning with Python and TensorFlow/Keras to adaptively guide learners.

---

The tutor suggests the next study action (e.g., practice, revision, new topic) for a student based on their profile and recent scores. The model is trained using a Deep Q-Network (DQN) on a custom Gym-style environment derived from the Students Performance dataset.

---

- **StudentsPerformance.csv** (Kaggle)
- Key features: gender, parental education, lunch, test preparation, math/reading/writing scores.

---


- Python 3.9+
- TensorFlow / Keras
- OpenAI Gym (custom environment)
- NumPy, Pandas, Matplotlib, Seaborn

---


1. Preprocess data (one-hot encoding + score normalization).  
2. Create a custom Gym-like environment where the state is a student profile and actions are study-path recommendations.  
3. Train a DQN agent (with replay buffer and target network).  
4. Evaluate the agent and produce human-readable recommendations.

---


