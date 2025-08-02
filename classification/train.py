import os, torch, pandas as pd, numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    df = pd.read_excel("classification/dataset.xlsx").dropna()
    return df['query'].astype(str).tolist(), df['label'].astype(int).tolist()

def get_embeddings(queries):
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    mdl = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
    mdl.eval()
    inputs = tok(queries, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        cls = mdl(**inputs).last_hidden_state[:,0,:].cpu().numpy()
    return cls

class MBA_RAG_Bandit:
    def __init__(self, n_arms, dim, alpha=1.0, epsilon=0.1):
        self.n_arms = n_arms
        self.alpha = alpha
        self.epsilon = epsilon
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0]*n_arms

    def select_arm(self, x):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        scores = []
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            score = x @ theta + self.alpha * np.sqrt(x @ A_inv @ x.T)
            scores.append(score)
        return int(np.argmax(scores))

    def update(self, arm, x, reward):
        self.counts[arm] += 1
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

if __name__ == "__main__":
    queries, labels = load_data()
    X = get_embeddings(queries)
    X_train, X_test, y_train, y_test, q_train, q_test = train_test_split(
        X, labels, queries, test_size=0.3, stratify=labels, random_state=42
    )

    bandit = MBA_RAG_Bandit(n_arms=2, dim=X.shape[1], alpha=1.0, epsilon=0.1)

    # Online training via imitation (using known labels as proxy)
    for x, y in zip(X_train, y_train):
        arm = bandit.select_arm(x)
        reward = 1.0 if arm == y else 0.0
        bandit.update(arm, x, reward)

    # Evaluation + predictions
    results = []
    correct = 0
    for x, y, q in zip(X_test, y_test, q_test):
        arm = bandit.select_arm(x)
        correct += int(arm == y)
        results.append({'query': q, 'predicted': arm, 'true': y, 'correct': int(arm == y)})

    accuracy = correct / len(y_test)
    print(f"Test Accuracy (proxy reward): {accuracy:.4f}")

    df = pd.DataFrame(results)
    os.makedirs("classification", exist_ok=True)
    df.to_excel("classification/mba_rag_predictions.xlsx", index=False)
    print("Saved MBA-RAG predictions to classification/mba_rag_predictions.xlsx")
