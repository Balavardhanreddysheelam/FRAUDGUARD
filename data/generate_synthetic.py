import pandas as pd
import random
import json
from datetime import datetime, timedelta

def generate_synthetic_data(num_rows=50000):
    print(f"Generating {num_rows} rows of synthetic financial QA data...")
    
    data = []
    
    fraud_scenarios = [
        ("I see a transaction I didn't make.", "Please freeze your card immediately through the app and contact support."),
        ("Why was my card declined?", "Your card may have been declined due to unusual activity. Please verify the transaction."),
        ("Is this email from you?", "We never ask for your password via email. This is likely a phishing attempt."),
        ("My wallet was stolen.", "Cancel all your cards immediately and file a police report."),
        ("What is this charge for 'Unknown Service'?", "This looks like a subscription service. Did you sign up for a free trial recently?")
    ]
    
    legit_scenarios = [
        ("What is my balance?", "Your current balance is available on the dashboard."),
        ("How do I transfer money?", "Go to the 'Transfer' tab and select the recipient."),
        ("Can I increase my credit limit?", "You can request a credit limit increase in the 'Settings' menu."),
        ("Where is the nearest ATM?", "Use the 'Find ATM' feature in the app to locate one near you."),
        ("How do I change my PIN?", "You can change your PIN at any ATM or through the app settings.")
    ]
    
    for i in range(num_rows):
        is_fraud_related = random.random() < 0.3
        
        if is_fraud_related:
            q, a = random.choice(fraud_scenarios)
            label = 1
        else:
            q, a = random.choice(legit_scenarios)
            label = 0
            
        # Add some noise/variation
        amount = round(random.uniform(10.0, 5000.0), 2)
        merchant = f"Merchant_{random.randint(1, 1000)}"
        
        data.append({
            "transaction_id": f"TXN_{i}",
            "amount": amount,
            "merchant": merchant,
            "user_question": q,
            "model_answer": a,
            "is_fraud_risk": label,
            "timestamp": (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat()
        })
        
    df = pd.DataFrame(data)
    output_path = "data/synthetic_financial_qa.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()
