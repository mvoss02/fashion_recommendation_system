from typing import List

class DataLoader:
    TRAIN_START_DATE: str = '2018-09-20'
    TEST_START_DATE: str = '2020-06-20'
    
    
class Variables:
    AGE_BINS: List[int] = [0, 20, 40, 60, 80, 100]
    AGE_LABELS: List[int] = ['0-20', '20-40', '40-60', '60-80', '80-100']
    ARTICLE_VARIABLES: List[str] = ['article_id', 'product_type_name', 'product_group_name', 'colour_group_name',
                                          'department_name', 'index_name', 'section_name', 'garment_group_name']
    CUSTOMER_VARIABLES: List[str] = ['customer_id', 'FN', 'Active', 'club_member_status',
                                           'fashion_news_frequency', 'age', 'postal_code']
    ALL_VARIABLES: List[str] = ARTICLE_VARIABLES + CUSTOMER_VARIABLES