"""Each person's own order of cards and lists: kept on the Hub per person, one screen at a time."""

from __future__ import annotations


def test_orders_are_per_person_and_per_screen(client, admin):
    assert client.get("/api/me/orders", headers=admin).json() == {}

    saved = client.put("/api/me/orders/review.projects", json={"ids": ["b", "a", "b", "c"]}, headers=admin).json()
    assert saved == {"review.projects": ["b", "a", "c"]}  # an id twice keeps its first place
    client.put("/api/me/orders/nodes", json={"ids": ["n2", "n1"]}, headers=admin)
    assert client.get("/api/me/orders", headers=admin).json() == {"review.projects": ["b", "a", "c"], "nodes": ["n2", "n1"]}

    # another person has none of them
    client.post("/api/users", json={"username": "kim", "password": "kim-orders-pw"}, headers=admin)
    token = client.post("/api/auth/login", json={"username": "kim", "password": "kim-orders-pw"}).json()["token"]
    kim = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/me/orders", headers=kim).json() == {}

    # an empty order: back to the screen's own
    assert client.put("/api/me/orders/nodes", json={"ids": []}, headers=admin).json() == {"review.projects": ["b", "a", "c"]}
    assert client.get("/api/me/orders").status_code == 401
