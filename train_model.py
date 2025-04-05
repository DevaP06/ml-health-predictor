model_data = {
    "model": model,
    "features": X.columns.tolist()
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)
