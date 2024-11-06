from modules.data_loader import load_data
import time
from modules.preprocess_data import preprocess
from modules.train import run_training
from config.config import Config
import pickle

def run_all():
    # Start the timer
    start_time = time.time()
    
    # Load/Extract the data
    print('Loading the data...')
    train_df, test_df, article_df = load_data()
    
    # Calculate the time taken and print it
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Done loading the data. Time taken: {elapsed_time:.2f} seconds!')
    
    # Preprocess the data to a TF Dataset
    print('Preprocessing the data...')

    preprocessed_hm_data = preprocess(train_df, test_df, article_df, batch_size=Config.batch_size)
    
    # Calculate the time taken and print it
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Done pre-processing the data. Time taken: {elapsed_time:.2f} seconds!')

    # Train
    print('Starting to train the model...')
    return run_training(preprocessed_hm_data)


if __name__ == '__main__':
    history = run_all()
    results = history.history
    
    pickle.dump(results, open('./data/results/final_results.p', 'wb'))