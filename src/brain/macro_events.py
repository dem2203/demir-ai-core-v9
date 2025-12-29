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
    # MAJOR EVENTS 2025 (Real Calendar)
    # Format: (month, day, hour_utc, event_name, impact%)
    MAJOR_EVENTS = [
        # Ocak 2025
        (1, 3, 13, "NFP Report", 75),
        (1, 10, 13, "CPI Data", 70),
        (1, 29, 19, "FED Rate Decision", 85),  # FOMC - Ocak
        
        # Şubat 2025
        (2, 7, 13, "NFP Report", 75),
        (2, 12, 13, "CPI Data", 70),
        
        # Mart 2025
        (3, 7, 13, "NFP Report", 75),
        (3, 12, 12, "CPI Data", 70),
        (3, 19, 18, "FED Rate Decision", 85),  # FOMC - Mart
        
        # Nisan 2025
        (4, 4, 13, "NFP Report", 75),
        (4, 10, 12, "CPI Data", 70),
        
        # Mayıs 2025
        (5, 2, 13, "NFP Report", 75),
        (5, 7, 18, "FED Rate Decision", 85),  # FOMC - Mayıs
        (5, 13, 12, "CPI Data", 70),
        
        # Haziran 2025
        (6, 6, 13, "NFP Report", 75),
        (6, 11, 12, "CPI Data", 70),
        (6, 18, 18, "FED Rate Decision", 85),  # FOMC - Haziran
    ]
    
    upcoming = []
    now = datetime.utcnow()  # UTC time for accurate comparison
    
    for month, day, hour, event_name, impact in MAJOR_EVENTS:
        try:
            event_time = datetime(now.year, month, day, hour)
            
            # Skip past events
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
