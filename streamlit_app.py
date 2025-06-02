import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from snowflake.snowpark

cnx = st.connection("snowflake")
session = cnx.session()

df = session.table("DF_MODEL_INPUT")
st.write(df)

df_model_input = df.to_pandas()

if not df_model_input.empty:
    # Select features for segmentation
    # Example: Age, MonetaryValue (LTV), Frequency, AvgSessionDuration, TotalPageviews, AOV
    segmentation_features_list = ['Age', 'MonetaryValue', 'Frequency', 'AOV', 'AvgSessionDuration', 'TotalPageviews', 'DaysSinceRegistration']
    # Add categorical features like 'Country', 'DeviceCategory' after one-hot encoding
    
    # Ensure only existing columns are selected
    available_segmentation_features = [col for col in segmentation_features_list if col in df_model_input.columns]
    
    # Separate numeric and categorical features for preprocessing
    numeric_features_segment = df_model_input[available_segmentation_features].select_dtypes(include=np.number).columns.tolist()
    categorical_features_segment = ['Country', 'DeviceCategory'] # Example
    categorical_features_segment = [col for col in categorical_features_segment if col in df_model_input.columns]

    # Create preprocessing pipelines for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Handles any remaining NaNs
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)) #Set sparse_output to False
    ])

    # Create a column transformer to apply transformations to the correct columns
    preprocessor_segment = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features_segment),
            ('cat', categorical_transformer, categorical_features_segment)
        ], remainder='drop') # drop other columns not specified

    # Create the full pipeline with preprocessing and K-Means
    # We need to fit_transform the preprocessor first, then apply K-Means
    
    df_segment_ready = df_model_input.copy()
    
    # Check if there's data to process
    if not df_segment_ready.empty and (numeric_features_segment or categorical_features_segment) :
        # Apply preprocessing
        processed_data_segment = preprocessor_segment.fit_transform(df_segment_ready)
        
        # Convert to DataFrame to inspect (optional)
        # Get feature names after one-hot encoding
        ohe_feature_names = []
        if categorical_features_segment: # Check if the list is not empty
             # Access the OneHotEncoder correctly if it's part of the pipeline
            if 'cat' in preprocessor_segment.named_transformers_ and hasattr(preprocessor_segment.named_transformers_['cat'], 'named_steps') and 'onehot' in preprocessor_segment.named_transformers_['cat'].named_steps:
                ohe_step = preprocessor_segment.named_transformers_['cat'].named_steps['onehot']
                if hasattr(ohe_step, 'get_feature_names_out'):
                     ohe_feature_names = ohe_step.get_feature_names_out(categorical_features_segment)
                else: # Fallback for older scikit-learn versions or if it's not fitted yet (though fit_transform is called)
                    # This part might be tricky if not fitted or if structure differs.
                    # For demonstration, we'll assume it can be accessed.
                    # In a real scenario, ensure the transformer is fitted.
                    pass # Simpler to just use the transformed numpy array for K-Means

        
        feature_names_processed = numeric_features_segment + list(ohe_feature_names)
        
        # If processed_data_segment is a NumPy array and you want it as a DataFrame:
        # df_processed_segment = pd.DataFrame(processed_data_segment, columns=feature_names_processed, index=df_segment_ready.index)
        # print("\nProcessed Data for Segmentation (sample):")
        # print(df_processed_segment.head())

        # --- Determine optimal K using the Elbow Method ---
        if processed_data_segment.shape[0] > 1 and processed_data_segment.shape[1] > 0: # Check if data exists
            inertia = []
            k_range = range(1, 11) # Test K from 1 to 10
            # Ensure K is not greater than the number of samples
            max_k = min(10, processed_data_segment.shape[0])
            if max_k < 1 : max_k = 1 # At least 1 cluster
            k_range = range(1, max_k +1)


            for k in k_range:
                if k == 0: continue # Skip k=0
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(processed_data_segment)
                inertia.append(kmeans.inertia_)

            if inertia: # Check if inertia list is populated
                plt.figure(figsize=(8, 5))
                plt.plot(k_range, inertia, marker='o')
                plt.title('Elbow Method for Optimal K')
                plt.xlabel('Number of Clusters (K)')
                plt.ylabel('Inertia')
                plt.grid(True)
                plt.show() # Display the plot

                # Choose K (e.g., K=4 based on the elbow plot)
                OPTIMAL_K = 4 # You'd typically pick this from the plot
                if OPTIMAL_K > processed_data_segment.shape[0]: # Cannot have more clusters than samples
                    OPTIMAL_K = processed_data_segment.shape[0]

                if OPTIMAL_K > 0:
                    kmeans_final = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
                    df_model_input['Segment'] = kmeans_final.fit_predict(processed_data_segment)

                    print(f"\nCustomers Segmented (K={OPTIMAL_K}):")
                    print(df_model_input[['Segment'] + available_segmentation_features].head())

                    # Analyze Segments
                    segment_summary = df_model_input.groupby('Segment')[available_segmentation_features].mean()
                    print("\nSegment Summary (Means):")
                    print(segment_summary)

                    # Interpreting segments:
                    # Segment 0: Might be 'Low Value', low frequency/monetary
                    # Segment 1: Might be 'High Value / Loyal', high frequency/monetary
                    # Segment 2: Might be 'New Customers', high recency, moderate other values
                    # Segment 3: Might be 'At Risk / Dormant', very high recency (long ago), low engagement
                    # The interpretation depends heavily on the mean values for each feature in each segment.
                    # 'Rich'/'poor' is hard to infer directly without income data. We use 'MonetaryValue' as a proxy for spending.
                    # Geolocation (Country) was one-hot encoded; its influence can be seen by looking at the original data for each segment.
                    # Age segmentation is directly part of the features.

                    # Example: Labeling segments (conceptual)
                    # This requires manual analysis of `segment_summary`
                    segment_labels = {
                        0: "Potentially Low Value",
                        1: "Potentially High Value / Active",
                        2: "Potentially New / Occasional",
                        3: "Potentially At Risk / Less Engaged"
                    }
                    # Ensure labels cover all found segments if OPTIMAL_K changes
                    df_model_input['SegmentLabel'] = df_model_input['Segment'].map(lambda x: segment_labels.get(x, f"Segment {x}"))
                    print("\nSegment Labels:")
                    print(df_model_input[['Segment', 'SegmentLabel'] + available_segmentation_features].head())
                else:
                    print("Not enough data or clusters to perform segmentation.")
            else:
                print("Not enough data points to perform K-Means inertia calculation.")
        else:
            print("No data available for K-Means or no features selected.")
    else:
        print("df_segment_ready is empty or no features selected for segmentation.")
else:
    print("df_model_input is empty, skipping segmentation.")

