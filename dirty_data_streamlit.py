import streamlit as st
import pandas as pd
import numpy as np
import datetime

# Step 1: Create Synthetic Shoe Sales Data
st.header("📦 Step 1: Create Synthetic Shoe Sales Data")

if st.button("🎲 Generate Synthetic Data"):
    # ✅ Clear everything except synthetic_df
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    np.random.seed(42)
    num_rows = 250

    product_ids = np.random.randint(10000, 99999, size=num_rows)
    age_group = np.random.choice(['Adult', 'Child'], size=num_rows)
    gender = np.random.choice(['Male', 'Female'], size=num_rows)
    product_types = [
        'RunMax Pro', 'SpeedStrider', 'UrbanFlex', 'QuickStep', 'SportEdge',
        'Everyday Comfort', 'Casual Breeze', 'SlipFlex', 'CityWalk', 'EasyStride',
        'SharpShoes', 'EliteStyle', 'Prestige Walk', 'ClassicFit', 'RefinedStep',
        'PowerRun', 'Agility Sprint', 'Endurance Racer', 'TrackMaster', 'ProFit',
        'SturdyStep', 'HeavyDuty Pro', 'TrekKing', 'MountainEdge', 'StormGuard',
        'FlexFit Trainer', 'PushLimit Trainer', 'FitForce', 'TrainerFlex', 'MaxPower Trainer'
    ]
    product_type = np.random.choice(product_types, size=num_rows)
    sizes = np.random.choice([7, 8, 9, 10, 11], size=num_rows)
    colors = np.random.choice(['Red', 'Blue', 'Black', 'Gray', 'White'], size=num_rows)
    prices = np.round(np.random.uniform(30, 150, size=num_rows), 2)
    stock_quantity = np.random.randint(10, 100, size=num_rows)

    df = pd.DataFrame({
        'Product ID': product_ids,
        'Age Group': age_group,
        'Gender': gender,
        'Product Type': product_type,
        'Size': sizes,
        'Color': colors,
        'Price': prices,
        'Stock Quantity': stock_quantity
    })

    # ✅ Store both clean and working copies
    st.session_state.synthetic_df = df
    st.session_state.dirty_df = df.copy()

    st.success("✅ Fresh synthetic dataset created and stored.")
    st.dataframe(df.head())

# Step 2: Add Dirt to Dataset
st.header("🧪 Step 2: Add Dirt to Dataset")

if 'synthetic_df' in st.session_state:
    with st.form("dirt_form"):
        missing_pct = st.slider("Missing %", 0, 50, 10)
        duplicate_pct = st.slider("Duplicate %", 0, 50, 10)
        inconsistency_pct = st.slider("Inconsistency %", 0, 50, 5)
        submitted = st.form_submit_button("🧼 Create Dirty Dataset")

    if submitted:
        def create_dirty_dataset(df, missing_percentage=10, duplicate_percentage=10, inconsistency_percentage=5):
            df = df.astype(str)

            missing_indices_price = np.random.choice(df.index, size=int(len(df) * (missing_percentage / 100)), replace=False)
            missing_indices_stock = np.random.choice(df.index, size=int(len(df) * (missing_percentage / 100)), replace=False)
            df.loc[missing_indices_price, 'Price'] = np.nan
            df.loc[missing_indices_stock, 'Stock Quantity'] = np.nan

            duplicates = df.sample(frac=duplicate_percentage / 100, random_state=42)
            df = pd.concat([df, duplicates], ignore_index=True)

            inconsistent_indices_gender = np.random.choice(df.index, size=int(len(df) * (inconsistency_percentage / 100)), replace=False)
            df.loc[inconsistent_indices_gender, 'Gender'] = np.random.choice(['male', 'FEMALE'], size=len(inconsistent_indices_gender))

            return df

        dirty_df = create_dirty_dataset(
            st.session_state.synthetic_df,
            missing_percentage=missing_pct,
            duplicate_percentage=duplicate_pct,
            inconsistency_percentage=inconsistency_pct
        )

        st.session_state.dirty_df = dirty_df
        dirty_df.to_csv("dirty_data.csv", index=False)
        st.success("✅ Dirty dataset created and saved as dirty_data.csv")
        st.dataframe(dirty_df.head())
else:
    st.warning("⚠️ Please generate the synthetic dataset first.")

# Step 3: Analyze Dirty Dataset
st.header("📊 Step 3: Analyze Dirty Dataset")

if 'dirty_df' in st.session_state:
    def analyze_dirty_data(df):
        rows_and_columns = f"Number of Rows: {df.shape[0]}\nNumber of Columns: {df.shape[1]}"
        duplicates = df[df.duplicated()]
        duplicates_count = f"Number of Duplicate Rows: {duplicates.shape[0]}"
        missing_data = df.isnull().sum()

        current_data_types = df.dtypes
        recommended_types = {
            'Product ID': 'int64',
            'Age Group': 'object',
            'Gender': 'object',
            'Product Type': 'object',
            'Size': 'int64',
            'Color': 'object',
            'Price': 'float64',
            'Stock Quantity': 'int64'
        }

        analysis_summary = pd.DataFrame({
            'Column': df.columns,
            'Missing Data': missing_data,
            'Current Data Type': current_data_types,
            'Recommended Data Type': [recommended_types.get(col, 'Unknown') for col in df.columns]
        })

        detailed_column_reports = []
        for col in df.columns:
            col_data = df[col].astype(str).fillna('')
            total = len(col_data)
            missing = col_data.isin(['', ' ', 'nan']).sum()
            unique_vals = col_data.nunique()
            value_counts = col_data.value_counts().head(5).to_dict()
            str_lengths = col_data.apply(len)
            min_len = str_lengths.min()
            max_len = str_lengths.max()
            avg_len = str_lengths.mean()

            lower_set = set([v.lower().strip() for v in col_data.unique()])
            casing_issues = unique_vals != len(lower_set)
            space_issues = any(col_data != col_data.str.strip())
            special_char_issues = col_data.str.contains(r'[^a-zA-Z0-9\\s]').any()

            # Adding Inconsistency Check
            inconsistency_check = f"Casing issues: {'Yes' if casing_issues else 'No'}, " \
                                  f"Space issues: {'Yes' if space_issues else 'No'}, " \
                                  f"Special character issues: {'Yes' if special_char_issues else 'No'}"

            report = f'''
Column: {col}
- Total entries: {total}
- Missing values (empty, space, 'nan'): {missing} ({(missing/total)*100:.2f}%)
- Unique values: {unique_vals}
- Top 5 most frequent values: {value_counts}
- String length: min = {min_len}, max = {max_len}, avg = {avg_len:.2f}
- Inconsistency Check: {inconsistency_check}
            '''.strip()

            detailed_column_reports.append(report)

        analysis_info = {
            'rows_and_columns': rows_and_columns,
            'duplicates_count': duplicates_count,
            'missing_data': missing_data,
            'detailed_reports': detailed_column_reports
        }

        return analysis_summary, analysis_info

    analysis_result, analysis_info = analyze_dirty_data(st.session_state.dirty_df)

    st.subheader("🧾 Data Summary")
    st.text(analysis_info['rows_and_columns'])
    st.text(analysis_info['duplicates_count'])

    st.subheader("📋 Column Summary Table")
    st.dataframe(analysis_result)

    st.subheader("🔎 Detailed Per-Column Reports")
    for report in analysis_info['detailed_reports']:
        st.markdown(f"```\n{report}\n```")
else:
    st.warning("⚠️ Dirty dataset not found. Please generate it first.")

# Step 4: Replace NaN with -99999
st.header("🔧 Step 4: Replace NaN with -99999")

st.markdown("> This step replaces all NaN and blank values with **-99999** to allow safe data type conversion (you can't convert `NaN` directly to float or int).")

if st.button("🔄 Replace NaNs with -99999"):
    def replace_all_nans_with_placeholder(df):
        df = df.copy()
        df = df.replace(['', ' ', 'nan', 'NaN', pd.NA], pd.NA)
        df = df.fillna(-99999)
        return df

    st.session_state.dirty_df = replace_all_nans_with_placeholder(st.session_state.dirty_df)
    st.success("✅ All NaNs and blanks replaced with -99999.")
    st.dataframe(st.session_state.dirty_df.head())

# Step 5: Change Data Types Interactively
if 'dirty_df' in st.session_state:
    st.header("🌀 Step 5: Change Data Types Interactively")

    def change_data_types(df):
        df = df.copy()
        recommended_types = {
            'Product ID': 'int64',
            'Age Group': 'object',
            'Gender': 'object',
            'Product Type': 'object',
            'Size': 'int64',
            'Color': 'object',
            'Price': 'float64',
            'Stock Quantity': 'int64'
        }

        st.markdown("📊 **Current Data Types of Columns:**")
        st.write(df.dtypes)

        if 'type_changes' not in st.session_state:
            st.session_state.type_changes = {}

        for column in df.columns:
            current_type = str(df[column].dtype)
            recommended = recommended_types.get(column, 'Unknown')
            default = recommended if recommended != 'Unknown' else current_type

            st.markdown(f"#### 🧾 Column: `{column}` — Current: `{current_type}`, Recommended: `{recommended}`")

            selected_type = st.selectbox(
                f"Choose new data type for `{column}`",
                options=['object', 'int64', 'float64'],
                index=['object', 'int64', 'float64'].index(default) if default in ['object', 'int64', 'float64'] else 0,
                key=f"select_{column}"
            )
            st.session_state.type_changes[column] = selected_type

        if st.button("✅ Apply Conversions"):
            success = True
            for col, new_type in st.session_state.type_changes.items():
                try:
                    df[col] = df[col].astype(new_type)
                    st.success(f"✅ Converted '{col}' to {new_type}")
                except Exception as e:
                    st.error(f"❌ Could not convert '{col}' to {new_type}: {e}")
                    success = False

            if success:
                st.session_state.dirty_df = df
                st.markdown("### ✅ Final Data Types After Conversion:")
                st.write(df.dtypes)

    change_data_types(st.session_state.dirty_df)
else:
    st.warning("⚠️ No dirty dataset found. Generate and clean NaNs first.")


# Final Message
today_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
total_cost = 7.50  # Static cost for demonstration purposes

# Sarcastic cost breakdown with humor
st.success(f"🎉 All data now cleaned!\n\nTotal cost is £{total_cost:.2f}. That's 50% more than a typical fiver gig, which would scrape 2 million pages from 2,000 sources in 2 hours... because, you know, quality costs extra. 😂\n\nHappy {today_date}!")

