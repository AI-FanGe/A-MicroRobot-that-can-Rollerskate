# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Xiaoxiaobai robot API: the on-board service behind the web UI.

Interfaces follow the PrimeBot (Q1) SDK conventions: McAction modes with a transition
table, prioritised velocity input sources with a watchdog, and CommonResponse-style
replies (header.code, state.value, state.reason, message, task_id).
"""

API_VERSION = "1.0.0"
