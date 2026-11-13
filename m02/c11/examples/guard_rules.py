from _shop import load_orders

WRITE_TOOLS = {"cancel_order"}
orders = load_orders()


def guard(customer_id, name, order_id, confirmed=False):
    if order_id not in orders.index:
        return "error: order_not_found"
    if orders.loc[order_id, "customer_id"] != customer_id:
        return "blocked: order belongs to another customer"
    if name in WRITE_TOOLS and not confirmed:
        return "pending: ask the customer to confirm in the app"
    return "allowed"


requests = [
    ("C04481", "cancel_order", "AT047838", False),  # đúng yêu cầu mà ghi chú độc hại muốn
    ("C04481", "get_order_status", "AT047838", False),  # đọc đơn của người khác cũng là rò rỉ
    ("C04481", "cancel_order", "AT052290", False),
    ("C04481", "cancel_order", "AT052290", True),
    ("C01024", "get_order_status", "AT047838", False),
]
for customer, name, oid, confirmed in requests:
    print(f"{customer} {name:<17}{oid} confirmed={confirmed!s:<6}", guard(customer, name, oid, confirmed))
