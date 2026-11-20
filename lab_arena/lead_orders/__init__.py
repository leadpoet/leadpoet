"""Gamma lead orders: the public, money-deciding side (docs/gamma-lead-orders.md).

Everything here is inert unless ``LEADPOET_LEAD_ORDERS_MODE`` names a mode
other than ``off`` (the default). The private lead-finding runner talks to the
gateway routes in ``gateway/api/lead_orders.py``; this package holds the
policy record, money arithmetic, Gamma reader, key ranking, broker, receipts
and the reward lane.
"""
