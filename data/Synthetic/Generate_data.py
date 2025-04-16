import random

import numpy as np
import pandas as pd

SIZE_CONSTANT = 10000

num_samples = 100 * SIZE_CONSTANT

synthetic_data = {
    'income': [random.uniform(20000, 100000) for _ in range(num_samples)],
    'name_email_similarity': [random.uniform(0, 1) for _ in range(num_samples)],
    'prev_address_months_count': [random.randint(0, 120) for _ in range(num_samples)],
    'current_address_months_count': [random.randint(0, 120) for _ in range(num_samples)],
    'customer_age': [random.randint(18, 90) for _ in range(num_samples)],
    'days_since_request': [random.randint(0, 365) for _ in range(num_samples)],
    'intended_balcon_amount': [random.uniform(1000, 50000) for _ in range(num_samples)],
    'zip_count_4w': [random.randint(0, 100) for _ in range(num_samples)],
    'fraud_bool': [random.choice([0, 1]) for _ in range(num_samples)]
}

fraud_data = pd.DataFrame(synthetic_data).head(20  * SIZE_CONSTANT)
fraud_data_f = pd.DataFrame(synthetic_data).tail(80  * SIZE_CONSTANT)


fraud_data_f['income'] *= np.random.uniform(1.05, 1.15, size=len(fraud_data_f))

fraud_data_f['name_email_similarity'] *= np.random.uniform(0.9, 0.95, size=len(fraud_data_f))
fraud_data_f['name_email_similarity'] = fraud_data_f['name_email_similarity'].clip(0, 1)

fraud_data_f['customer_age'] += np.random.randint(1, 5, size=len(fraud_data_f))
fraud_data_f['customer_age'] = fraud_data_f['customer_age'].clip(18, 100)