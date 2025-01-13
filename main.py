from src.preprocessing import load_data, clean_data, split_data
#from src.model_training import train_model, save_model
#from src.evaluation import evaluate_model

def main():
    data = load_data('data/raw/train.csv')
    data = clean_data(data)
    split_data(data,'data/processed/train.csv','data/processed/test.csv')
    #model = train_model(X_train, y_train)
    #save_model(model, "models/house_price_model")
    #evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
