from fastapi import FastAPI

# Create FastAPI app instance
app = FastAPI(title="E-Commerce Store API")


# ======================================================
# Product Data (Temporary In-Memory Database)
# ======================================================

products = [
    {"id": 1, "name": "Wireless Mouse", "price": 599, "category": "Electronics", "in_stock": True},
    {"id": 2, "name": "Notebook", "price": 99, "category": "Stationery", "in_stock": True},
    {"id": 3, "name": "Pen Set", "price": 49, "category": "Stationery", "in_stock": True},
    {"id": 4, "name": "USB Cable", "price": 199, "category": "Electronics", "in_stock": False},
    {"id": 5, "name": "Laptop Stand", "price": 1299, "category": "Electronics", "in_stock": True},
    {"id": 6, "name": "Mechanical Keyboard", "price": 2499, "category": "Electronics", "in_stock": True},
    {"id": 7, "name": "Webcam", "price": 1899, "category": "Electronics", "in_stock": False},
]


# ======================================================
# Root Endpoint
# ======================================================

@app.get("/")
def home():
    """
    Basic welcome route to confirm API is running.
    """
    return {"message": "Welcome to My E-Commerce API 🚀"}


# ======================================================
# Q1 – Get All Products
# ======================================================

@app.get("/products")
def get_all_products():
    """
    Returns all products along with total count.
    """
    return {
        "products": products,
        "total_products": len(products)
    }


# ======================================================
# Q2 – Filter Products by Category
# ======================================================

@app.get("/products/category/{category_name}")
def get_products_by_category(category_name: str):
    """
    Returns products belonging to a specific category.
    """
    filtered_products = [
        product for product in products
        if product["category"].lower() == category_name.lower()
    ]

    if not filtered_products:
        return {"error": "No products found in this category"}

    return {
        "category": category_name,
        "products": filtered_products,
        "total": len(filtered_products)
    }


# ======================================================
# Q3 – Get Only In-Stock Products
# ======================================================

@app.get("/products/instock")
def get_instock_products():
    """
    Returns only products that are currently in stock.
    """
    available_products = [
        product for product in products
        if product["in_stock"]
    ]

    return {
        "in_stock_products": available_products,
        "count": len(available_products)
    }


# ======================================================
# Q4 – Store Summary
# ======================================================

@app.get("/store/summary")
def get_store_summary():
    """
    Provides overall store statistics:
    - Total products
    - In-stock count
    - Out-of-stock count
    - Unique categories
    """

    total_products = len(products)
    in_stock_count = len([p for p in products if p["in_stock"]])
    out_of_stock_count = total_products - in_stock_count
    unique_categories = list(set(p["category"] for p in products))

    return {
        "store_name": "My E-Commerce Store",
        "total_products": total_products,
        "in_stock": in_stock_count,
        "out_of_stock": out_of_stock_count,
        "categories": unique_categories
    }


# ======================================================
# Q5 – Search Products (Case-Insensitive)
# ======================================================

@app.get("/products/search/{keyword}")
def search_products(keyword: str):
    """
    Searches products by name.
    Case-insensitive matching.
    """

    matched_products = [
        product for product in products
        if keyword.lower() in product["name"].lower()
    ]

    if not matched_products:
        return {"message": "No products matched your search"}

    return {
        "search_keyword": keyword,
        "results": matched_products,
        "total_matches": len(matched_products)
    }


# ======================================================
# BONUS – Best Deal & Premium Pick
# ======================================================

@app.get("/products/deals")
def get_product_deals():
    """
    Returns:
    - Cheapest product (Best Deal)
    - Most expensive product (Premium Pick)
    """

    cheapest_product = min(products, key=lambda p: p["price"])
    most_expensive_product = max(products, key=lambda p: p["price"])

    return {
        "best_deal": cheapest_product,
        "premium_pick": most_expensive_product
    }