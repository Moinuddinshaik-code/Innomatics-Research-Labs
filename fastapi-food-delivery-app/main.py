from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field
from typing import Optional
import math

app = FastAPI()

# ---------------- DATA ----------------
menu = [
    {"id": 1, "name": "Pizza", "price": 250, "category": "Food", "is_available": True},
    {"id": 2, "name": "Burger", "price": 150, "category": "Food", "is_available": True},
    {"id": 3, "name": "Pasta", "price": 200, "category": "Food", "is_available": True},
    {"id": 4, "name": "Coke", "price": 50, "category": "Drink", "is_available": True},
    {"id": 5, "name": "Ice Cream", "price": 120, "category": "Dessert", "is_available": True},
]

orders = []
cart = []
order_counter = 1
menu_counter = 6

# ---------------- MODELS ----------------
class OrderRequest(BaseModel):
    customer_name: str = Field(min_length=2)
    item_id: int = Field(gt=0)
    quantity: int = Field(gt=0, le=20)
    address: str = Field(min_length=5)

class NewMenuItem(BaseModel):
    name: str = Field(min_length=2)
    price: int = Field(gt=0)
    category: str = Field(min_length=2)
    is_available: bool = True

class CheckoutRequest(BaseModel):
    customer_name: str = Field(min_length=2)
    address: str = Field(min_length=5)

# ---------------- HELPERS ----------------
def find_item(item_id):
    for item in menu:
        if item["id"] == item_id:
            return item
    return None

def calculate_total(price, quantity):
    return price * quantity

def filter_menu_logic(category=None, max_price=None, is_available=None):
    result = menu
    if category is not None:
        result = [i for i in result if i["category"].lower() == category.lower()]
    if max_price is not None:
        result = [i for i in result if i["price"] <= max_price]
    if is_available is not None:
        result = [i for i in result if i["is_available"] == is_available]
    return result

# ---------------- GET ----------------
@app.get("/")
def home():
    return {"message": "Welcome to Food Delivery API"}

@app.get("/menu")
def get_menu():
    return {"menu": menu, "total": len(menu)}

@app.get("/menu/summary")
def summary():
    available = len([i for i in menu if i["is_available"]])
    categories = list(set(i["category"] for i in menu))
    return {
        "total": len(menu),
        "available": available,
        "categories": categories
    }

@app.get("/menu/filter")
def filter_menu(
    category: Optional[str] = None,
    max_price: Optional[int] = None,
    is_available: Optional[bool] = None
):
    result = filter_menu_logic(category, max_price, is_available)
    return {"items": result, "count": len(result)}

@app.get("/menu/search")
def search(keyword: str):
    result = [i for i in menu if keyword.lower() in i["name"].lower() or keyword.lower() in i["category"].lower()]
    return {"results": result, "total_found": len(result)}

@app.get("/menu/sort")
def sort_menu(sort_by: str = "price", order: str = "asc"):
    if sort_by not in ["price", "name", "category"]:
        raise HTTPException(400, "Invalid sort field")

    reverse = True if order == "desc" else False
    sorted_menu = sorted(menu, key=lambda x: x[sort_by], reverse=reverse)

    return {"sorted": sorted_menu}

@app.get("/menu/page")
def paginate(page: int = 1, limit: int = 3):
    start = (page - 1) * limit
    total = len(menu)
    total_pages = math.ceil(total / limit)

    return {
        "page": page,
        "total_pages": total_pages,
        "items": menu[start:start+limit]
    }

@app.get("/menu/browse")
def browse(
    keyword: Optional[str] = None,
    sort_by: str = "price",
    order: str = "asc",
    page: int = 1,
    limit: int = 3
):
    result = menu

    # filter
    if keyword:
        result = [i for i in result if keyword.lower() in i["name"].lower()]

    # sort
    reverse = True if order == "desc" else False
    result = sorted(result, key=lambda x: x[sort_by], reverse=reverse)

    # pagination
    start = (page - 1) * limit
    total = len(result)

    return {
        "total": total,
        "items": result[start:start+limit]
    }

@app.get("/menu/{item_id}")
def get_item(item_id: int):
    item = find_item(item_id)
    if not item:
        raise HTTPException(404, "Item not found")
    return item

# ---------------- POST ----------------
@app.post("/menu")
def add_item(item: NewMenuItem, response: Response):
    global menu_counter

    for i in menu:
        if i["name"].lower() == item.name.lower():
            raise HTTPException(400, "Duplicate item")

    new_item = item.dict()
    new_item["id"] = menu_counter
    menu.append(new_item)
    menu_counter += 1

    response.status_code = 201
    return new_item

@app.post("/orders")
def create_order(order: OrderRequest):
    global order_counter

    item = find_item(order.item_id)
    if not item:
        raise HTTPException(404, "Item not found")

    if not item["is_available"]:
        raise HTTPException(400, "Item not available")

    total = calculate_total(item["price"], order.quantity)

    new_order = {
        "order_id": order_counter,
        "customer": order.customer_name,
        "item": item["name"],
        "total": total
    }

    orders.append(new_order)
    order_counter += 1

    return new_order

# ---------------- PUT ----------------
@app.put("/menu/{item_id}")
def update_item(item_id: int, price: Optional[int] = None, is_available: Optional[bool] = None):
    item = find_item(item_id)
    if not item:
        raise HTTPException(404, "Item not found")

    if price is not None:
        item["price"] = price
    if is_available is not None:
        item["is_available"] = is_available

    return item

# ---------------- DELETE ----------------
@app.delete("/menu/{item_id}")
def delete_item(item_id: int):
    item = find_item(item_id)
    if not item:
        raise HTTPException(404, "Item not found")

    menu.remove(item)
    return {"message": "Deleted successfully"}

# ---------------- CART ----------------
@app.post("/cart/add")
def add_to_cart(item_id: int, quantity: int = 1):
    item = find_item(item_id)
    if not item:
        raise HTTPException(404, "Item not found")

    for c in cart:
        if c["item_id"] == item_id:
            c["quantity"] += quantity
            return {"message": "Updated cart"}

    cart.append({"item_id": item_id, "quantity": quantity})
    return {"message": "Added to cart"}

@app.get("/cart")
def view_cart():
    total = 0
    detailed = []

    for c in cart:
        item = find_item(c["item_id"])
        subtotal = item["price"] * c["quantity"]
        total += subtotal

        detailed.append({
            "name": item["name"],
            "quantity": c["quantity"],
            "subtotal": subtotal
        })

    return {"cart": detailed, "total": total}

@app.post("/cart/checkout")
def checkout(data: CheckoutRequest, response: Response):
    global order_counter

    if not cart:
        raise HTTPException(400, "Cart is empty")

    created_orders = []
    total = 0

    for c in cart:
        item = find_item(c["item_id"])
        subtotal = item["price"] * c["quantity"]

        order = {
            "order_id": order_counter,
            "customer": data.customer_name,
            "item": item["name"],
            "total": subtotal
        }

        created_orders.append(order)
        orders.append(order)
        total += subtotal
        order_counter += 1

    cart.clear()
    response.status_code = 201

    return {"orders": created_orders, "grand_total": total}