from sklearn.metrics import mean_squared_error, r2_score
def evaluate(x_data, y_data,model,name):
    y_data_pred = model.predict(x_data)
    print(f"{name} R2 Score: {r2_score(y_data, y_data_pred):.4f}")
    print(f"{name} MSE: , {mean_squared_error(y_data, y_data_pred)}")
