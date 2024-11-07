from typing import List

class DataLoader:
    TRAIN_START_DATE: str = '2019-09-20'
    TEST_START_DATE: str = '2020-06-20'
    
    
class Variables:
    AGE_BINS: List[int] = [0, 20, 40, 60, 80, 100]
    AGE_LABELS: List[int] = ['0-20', '20-40', '40-60', '60-80', '80-100']
    ARTICLE_VARIABLES: List[str] = ['article_id', 'product_type_name', 'product_group_name', 'colour_group_name', 'department_name', 'index_name', 'section_name', 'garment_group_name']
    CUSTOMER_VARIABLES_STR: List[str] = ['customer_id', 'club_member_status', 'fashion_news_frequency', 'age_bins', 'postal_code']
    CUSTOMER_VARIABLES_NUM: List[str] = ['FN', 'Active']
    ALL_CUSTOMER_VARIABLES: List[str] = CUSTOMER_VARIABLES_STR + CUSTOMER_VARIABLES_NUM
    ALL_VARIABLES: List[str] = ARTICLE_VARIABLES + CUSTOMER_VARIABLES_STR + CUSTOMER_VARIABLES_NUM
    

class Config:
    epochs: int = 3
    output_layer_size: int = 32
    hidden_layer_sizes: List[int] = []
    learning_rate: float = 0.001
    batch_size: int = 512
    dropout_rate: float = None
    l2_decay: float = None
    using_cross: str = 'No'
    split_type: str = 'Classic temporal'
    log_dir: str = './logs'
    embedding_dimension: int = 128
    