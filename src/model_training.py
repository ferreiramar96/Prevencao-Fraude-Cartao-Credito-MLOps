import sklearn

def train_model(model:sklearn, x_train, y_train):
    model.fit(x_train, y_train)
    return model