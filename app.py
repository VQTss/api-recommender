from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Đọc dữ liệu từ file CSV
file_path = './data/Modified_Product_Ratings_Beauty_Data.csv'
data = pd.read_csv(file_path)

# Chuẩn bị dữ liệu cho mô hình recommender
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data[['UserId', 'ProductId', 'Rating']], reader)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
trainset, testset = train_test_split(dataset, test_size=0.25)

# Huấn luyện mô hình SVD
algo = SVD()
algo.fit(trainset)

# Định nghĩa hàm gợi ý sản phẩm
def get_top_n_recommendations(user_id, n=10):
    # Lấy danh sách các sản phẩm mà người dùng chưa đánh giá
    all_product_ids = data['ProductId'].unique()
    rated_products = data[data['UserId'] == user_id]['ProductId']
    
    if rated_products.empty:
        print(f"UserId {user_id} does not exist in the dataset")
        return []
    
    unrated_products = [product for product in all_product_ids if product not in rated_products.values]
    
    if not unrated_products:
        print(f"UserId {user_id} has rated all products. Recommending popular products.")
        popular_products = data['ProductId'].value_counts().index[:n].tolist()
        return popular_products
    
    # Dự đoán xếp hạng cho các sản phẩm chưa đánh giá
    predictions = [algo.predict(user_id, product_id) for product_id in unrated_products]
    
    # Sắp xếp và lấy n sản phẩm hàng đầu
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    top_n_product_ids = [pred.iid for pred in top_n]
    
    return top_n_product_ids

# Định nghĩa FastAPI app
app = FastAPI()

# Định nghĩa lớp dữ liệu đầu vào
class UserData(BaseModel):
    UserId: int
    n: int = 10

# Định nghĩa endpoint cho API recommender
@app.post("/recommend", response_model=List[int])
def recommend(user_data: UserData):
    user_id = user_data.UserId
    n = user_data.n
    
    try:
        recommendations = get_top_n_recommendations(user_id, n)
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"No recommendations found for UserId {user_id}")
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return recommendations

# Chạy server FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
