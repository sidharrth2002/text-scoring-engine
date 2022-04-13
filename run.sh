source /Users/SidharrthNagappan/.virtualenvs/multitask-bert/bin/activate
cd ./api && uvicorn app.api:app --reload & cd client && streamlit run app.py