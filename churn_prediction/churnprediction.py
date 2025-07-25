
# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
def preprocess_data(df):
    df = df.copy()
    
    # Conv TotalCharges to numeric 
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    median_val = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_val)
    
    # Drop ID
    df = df.drop('customerID', axis=1)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    return df

processed_df = preprocess_data(df)

# Split data
X = processed_df.drop('Churn', axis=1)
y = processed_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\nROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    return y_pred, y_proba

y_pred, y_proba = evaluate_model(rf_model, X_test_scaled, y_test)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

#dashboard
app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H1("Customer Churn Prediction", style={'textAlign': 'center'}),
    
    html.Div([
        dcc.Tabs(id='tabs', value='tab-1', children=[
            dcc.Tab(label='Data Overview', value='tab-1'),
            dcc.Tab(label='Feature Analysis', value='tab-2'),
            dcc.Tab(label='Model Performance', value='tab-3'),
            dcc.Tab(label='Churn Predictions', value='tab-4')
        ]),
        html.Div(id='tabs-content')
    ])
])

@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3("Dataset Overview"),
            dcc.Graph(
                figure=px.histogram(
                    df, 
                    x='Churn', 
                    color='Churn', 
                    title='Churn Distribution'
                )
            ),
            dcc.Graph(
                figure=px.imshow(
                    processed_df.corr(),
                    title='Correlation Matrix',
                    height=800
                )
            )
        ])
        
    elif tab == 'tab-2':
        return html.Div([
            html.H3("Feature Importance"),
            dcc.Graph(
                figure=px.bar(
                    feature_importance.head(10),
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title='Top 10 Important Features'
                )
            ),
            html.H3("Numerical Feature Distribution"),
            dcc.Dropdown(
                id='num-feature-selector',
                options=[{'label': col, 'value': col} 
                         for col in ['tenure', 'MonthlyCharges', 'TotalCharges']],
                value='tenure'
            ),
            dcc.Graph(id='num-feature-plot')
        ])
        
    elif tab == 'tab-3':
        # Generate classification report data
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1-Score', 'ROC AUC'],
            'Value': [
                report['1']['precision'],
                report['1']['recall'],
                report['1']['f1-score'],
                roc_auc_score(y_test, y_proba)
            ]
        })
        
        return html.Div([
            html.H3("Model Evaluation Metrics"),
            dcc.Graph(
                figure=go.Figure(
                    data=go.Heatmap(
                        z=confusion_matrix(y_test, y_pred),
                        x=['Predicted No', 'Predicted Yes'],
                        y=['Actual No', 'Actual Yes'],
                        colorscale='Blues',
                        texttemplate="%{z}",
                        showscale=False
                    ),
                    layout=go.Layout(title='Confusion Matrix')
                )
            ),
            dcc.Graph(
                figure=px.bar(
                    metrics_df,
                    x='Metric',
                    y='Value',
                    title='Performance Metrics',
                    text_auto='.2f'
                )
            )
        ])
        
    elif tab == 'tab-4':
        # Create high-risk customers table
        test_df = X_test.copy()
        test_df['Churn_Probability'] = y_proba
        test_df['Actual_Churn'] = y_test.values
        test_df['Predicted_Churn'] = y_pred
        
        #  only important columns for display
        display_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_Probability', 'Actual_Churn', 'Predicted_Churn']
        high_risk = test_df.sort_values('Churn_Probability', ascending=False).head(10)[display_cols]
        
        # Format the probability 
        high_risk['Churn_Probability'] = high_risk['Churn_Probability'].apply(lambda x: f"{x:.2%}")
        high_risk['Actual_Churn'] = high_risk['Actual_Churn'].map({0: 'No', 1: 'Yes'})
        high_risk['Predicted_Churn'] = high_risk['Predicted_Churn'].map({0: 'No', 1: 'Yes'})
        
        return html.Div([
            html.H3("Churn Probability Distribution"),
            dcc.Graph(
                figure=px.histogram(
                    x=y_proba,
                    color=y_test.astype(str),
                    nbins=50,
                    title='Predicted Churn Probabilities',
                    labels={'x': 'Churn Probability', 'color': 'Actual Churn'}
                )
            ),
            html.H3("Top 10 High-Risk Customers"),
            html.Table([
                html.Thead(
                    html.Tr([html.Th(col) for col in high_risk.columns])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(high_risk.iloc[i][col]) for col in high_risk.columns
                    ]) for i in range(len(high_risk))
                ])
            ], style={'margin': 'auto', 'width': '80%', 'border': '1px solid black'})
        ])

@app.callback(
    Output('num-feature-plot', 'figure'),
    [Input('num-feature-selector', 'value')]
)
def update_num_feature_plot(selected_feature):
    fig = px.box(
        df, 
        x='Churn', 
        y=selected_feature, 
        color='Churn',
        title=f'{selected_feature} Distribution by Churn Status'
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)