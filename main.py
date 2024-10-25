from data_loader import load_data
import time

def run_all():
    # Start the timer
    start_time = time.time()
    
    # Load the data
    print('Loading the data...')
    df_train, df_test, df_article = load_data()
    
    # Calculate the time taken and print it
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Done loading the data. Time taken: {elapsed_time:.2f} seconds!')
    
    df_train.head(5)


if __name__ == '__main__':
    run_all()