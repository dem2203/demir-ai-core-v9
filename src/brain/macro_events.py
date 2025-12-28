# -*- coding: utf-8 -*-
"""
Macro Event Protection
======================
FED, NFP, CPI gibi high-impact olaylar için uyarı sistemi.
"""
from datetime import datetime, timedelta
from typing import List, Dict


def check_upcoming_events(hours_ahead: int = 12) -> List[Dict]:
    """
    Check for high-impact macro events in next N hours.
    
    Returns:
        List of events: [{'name': 'FED Rate', 'hours_until': 6, 'impact': 85}, ...]
    """
    # HARDCODED MAJOR EVENTS (Production: Use API like Trading Economics)
    MAJOR_EVENTS = [
        (12, 29, 14, "FED Rate Decision", 85),
        (1, 2, 13, "NFP Report", 75),
        (1, 5, 14, "CPI Data", 70),
    ]
    
    upcoming = []
    now = datetime.now()
    
    for month, day, hour, event_name, impact in MAJOR_EVENTS:
        try:
            event_time = datetime(now.year, month, day, hour)
            
            if event_time < now:
                continue
            
            hours_until = (event_time - now).total_seconds() / 3600
            
            if hours_until <= hours_ahead:
                upcoming.append({
                    'name': event_name,
                    'time': event_time.isoformat(),
                    'hours_until': int(hours_until),
                    'impact': impact
                })
        except:
            continue
    
    return upcoming
