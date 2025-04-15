import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
email_df=pd.read_csv(r"C:\Users\ketan\ketan_python\assignment\email_table.csv")
email_open_df=pd.read_csv(r"C:\Users\ketan\ketan_python\assignment\email_opened_table.csv")
email_link_click_df=pd.read_csv(r"C:\Users\ketan\ketan_python\assignment\link_clicked_table.csv")

#analysis data
# print(email_df.head().to_string())
# print()
# print(email_open_df.head().to_string())
# print()
# print(email_link_click_df.head().to_string())
# print()

#percentage of email open
open_mail_percentage=(len(email_open_df)/len(email_df))*100
print(f"{open_mail_percentage:.2f}% Open the mail.")

#percentage of email link click
click_link_percentage=(len(email_link_click_df)/len(email_df))*100
print(f"{click_link_percentage:.2f}% Click the link.")
print()

#merge dataset
email_df['Open']=email_df['email_id'].isin(email_open_df['email_id']).astype(int)
email_df['clicked'] = email_df['email_id'].isin(email_link_click_df['email_id']).astype(int)

#print the merged dataset
print("Dataset")
print(email_df.head().to_string())
print()

#encoding label data using LabelEncoder()
lc_text=LabelEncoder()
lc_version=LabelEncoder()
lc_weekday=LabelEncoder()
lc_Country=LabelEncoder()
email_df['email_text_enc']=lc_text.fit_transform(email_df['email_text'])
email_df['email_version_enc']=lc_version.fit_transform(email_df['email_version'])
email_df['weekday_enc']=lc_weekday.fit_transform(email_df['weekday'])
email_df['user_country_enc']=lc_Country.fit_transform(email_df['user_country'])

# print(email_df.head().to_string())

#give value to dependent and independent variable
x=email_df[['email_text_enc','email_version_enc','hour','weekday_enc','user_country_enc','user_past_purchases','Open']]
y=email_df['clicked']


#split data in train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


#apply RandomForest classification model
model=RandomForestClassifier()
model.fit(x_train,y_train)


# Taking user input
print("Email Text options:", list(lc_text.classes_))
email_text_input = input("Enter Email Text: ")
print()

print("Email Version options:", list(lc_version.classes_))
email_version_input = input("Enter Email Version: ")
print()

hour_input = int(input("Enter Hour (0 to 23): "))
print()

print("Weekday options:", list(lc_weekday.classes_))
weekday_input = input("Enter Weekday: ")
print()

print("User Country options:", list(lc_Country.classes_))
user_country_input = input("Enter User Country: ")
print()

user_past_purchases_input = int(input("Enter number of past purchases: "))
print()

open_input = int(input("Did user open email? Enter 1 for Yes, 0 for No: "))
print()

#encoded the user input
email_text_encoded = lc_text.transform([email_text_input])[0]
email_version_encoded = lc_version.transform([email_version_input])[0]
weekday_encoded = lc_weekday.transform([weekday_input])[0]
user_country_encoded = lc_Country.transform([user_country_input])[0]

#create dataframe of encoded user input
user_input_df = pd.DataFrame([[
    email_text_encoded,
    email_version_encoded,
    hour_input,
    weekday_encoded,
    user_country_encoded,
    user_past_purchases_input,
    open_input
]], columns=['email_text_enc','email_version_enc','hour','weekday_enc','user_country_enc','user_past_purchases','Open'])

#prediction from user input data
prediction = model.predict(user_input_df)[0]

#prediction result
print("\n✅ Prediction Result:")
if prediction == 1:
    print("User is likely to CLICK the link in the email.")
else:
    print("User is NOT likely to click the link in the email.")

#model accuracy
acc=model.score(x_test,y_test)*100
print(f"Accuracy: {acc}")

#Define classification report and confusion matrix
y_pred = model.predict(x_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()


# Predict probabilities on full dataset
probs = model.predict_proba(x)[:, 1]  # probability of class 1 (clicked)
email_df['predicted_prob'] = probs

# Top 20% users with highest predicted probability
top_20 = email_df.sort_values(by='predicted_prob', ascending=False).head(int(0.2 * len(email_df)))

# Calculate click-through rates
actual_ctr = email_df['clicked'].mean() * 100
model_ctr = top_20['clicked'].mean() * 100

print(f"Original campaign CTR: {actual_ctr:.2f}%")
print()
print(f"Model-targeted top 20% CTR: {model_ctr:.2f}%")
print()
print(f"🔼 Estimated CTR improvement: {model_ctr - actual_ctr:.2f}%")

#Graph
# Day of week vs Clicks
sns.barplot(data=email_df, x='weekday', y='clicked')
plt.title('Click Rate by Weekday')
plt.show()

# Personalization vs Clicks
sns.barplot(data=email_df, x='email_version', y='clicked')
plt.title('Click Rate by Email Version')
plt.show()

# Purchases vs Click Rate
email_df['purchase_bins'] = pd.cut(email_df['user_past_purchases'], bins=[-1, 0, 2, 5, 100], labels=['0', '1-2', '3-5', '6+'])
sns.barplot(data=email_df, x='purchase_bins', y='clicked')
plt.title('Click Rate by Past Purchases')
plt.show()



#Description
"""
Project: Email Marketing Campaign Optimization
Author: Ketan Chauhan
Goal: Analyze email campaign performance, build a model to predict link clicks,
      and provide insights to improve future email targeting.

Key Results:
- Open rate: 10.35%
- Click rate: 2.12%
- Model accuracy: 97.13%
- CTR improvement with model: 8.26%

Instructions:
1. Make sure CSV files are in the same directory.
2. Run this script using Python 3.11.0
"""
