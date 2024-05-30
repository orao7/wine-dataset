import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

class WineDecisionTreeModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = DecisionTreeClassifier(random_state=self.random_state)
        self.scaler = StandardScaler()
        self.load_data()
        self.prepare_data()

    def load_data(self):
        wine = load_wine()
        self.X = wine.data
        self.y = wine.target

    def prepare_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Decision Tree Accuracy: {accuracy:.2f}")

        cv_scores = cross_val_score(self.model, self.X, self.y, cv=5)
        print(f"Decision Tree Accuracy (K-Fold CV): {cv_scores.mean():.2f}")

    def visualize(self):
        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model, 
            filled=True, 
            feature_names=load_wine().feature_names, 
            class_names=load_wine().target_names
        )
        plt.show()

if __name__ == "__main__":
    model = WineDecisionTreeModel()
    model.train()
    model.evaluate()
    model.visualize()

