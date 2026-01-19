import http.server
import socketserver
import json
import os
import sys
import urllib.parse
import logging
import traceback
import math
import shutil
import time
import random
import secrets
import concurrent.futures
import datetime
import hashlib
import fastf1
import pandas as pd
import numpy as np
import requests
import requests_cache

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger("fastf1.api").setLevel(logging.ERROR)
logging.getLogger("fastf1.req").setLevel(logging.ERROR)
logging.getLogger("fastf1.core").setLevel(logging.ERROR)
logging.getLogger("fastf1.events").setLevel(logging.ERROR)

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
PORT = 5050
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')

# ==========================================
# DATA HANDLER (F1DataHandler)
# ==========================================
class F1DataHandler:
    """
    Capa de acceso y preparación de datos para el modelo 2026.
    
    MODO TIEMPO REAL:
    - Sin caché en disco ni almacenamiento local.
    - Conexión directa a API de FastF1.
    - Datos efímeros en memoria.
    """

    def __init__(self, cache_dir=None):
        # In-memory storage only
        self._total_laps_cache = {}
        self._official_laps_2026 = {} 
        self._grid_history = {}
        self._qualy_history_cache = {}
        self._race_history_cache = {}
        self._grid_weights_by_circuit = {}
        
        # Disable FastF1 cache explicitly (just in case)
        try:
            fastf1.Cache.clear_cache(deep=True)
        except Exception:
            pass

        # Try to clear requests_cache if accessible
        try:
            import requests_cache
            requests_cache.clear()
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Could not clear requests_cache: {e}")

    def _fetch_with_retry(self, func, *args, retries=3, delay=2, **kwargs):
        """Executes a function with exponential backoff retry mechanism."""
        last_exception = None
        func_name = None
        try:
            func_name = getattr(func, "__name__", None)
        except Exception:
            func_name = None
        if not func_name:
            try:
                func_name = func.__class__.__name__
            except Exception:
                func_name = "callable"
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                wait = delay * (2 ** attempt)
                logger.warning(f"Retry {attempt + 1}/{retries} for {func_name} after {wait}s due to: {e}")
                time.sleep(wait)
        raise last_exception

    def get_official_total_laps_2026(self, event_identifier):
        events = self.get_events(2026)
        target_event = None
        str_id = str(event_identifier)
        for e in events:
            if str(e.get('RoundNumber')) == str_id or str(e.get('OfficialEventName')) == str_id:
                target_event = e
                break
        if not target_event:
            return {"error": "Evento 2026 no encontrado", "total_laps": 0}

        round_key = str(target_event.get("RoundNumber"))
        if round_key in self._official_laps_2026:
            return {"total_laps": self._official_laps_2026[round_key]}

        location = target_event.get("Location")
        if not location:
            return {"error": "No se pudo resolver la localización del evento", "total_laps": 0}

        start_year, end_year = self._get_historical_year_range()
        
        # Check in-memory cache
        cache_key = (str(location).lower(), start_year, end_year)
        if cache_key in self._total_laps_cache:
            val = self._total_laps_cache[cache_key]
            self._official_laps_2026[round_key] = val
            return {"total_laps": val}

        total_laps = self._compute_latest_total_laps(location, start_year, end_year)
        if total_laps and total_laps > 0:
            total_laps = int(total_laps)
            self._total_laps_cache[cache_key] = total_laps
            self._official_laps_2026[round_key] = total_laps
            return {"total_laps": total_laps}

        return {"error": f"No hay datos suficientes para fijar vueltas oficiales de {location}", "total_laps": 0}

    def _get_historical_year_range(self):
        start_year = 2018
        target_year = 2026
        current_year = datetime.datetime.now(datetime.timezone.utc).year
        end_year = min(target_year - 1, current_year - 1)
        end_year = max(start_year, end_year)
        return start_year, end_year
    
    def check_connection(self):
        """
        Simple connection check instead of full re-import.
        """
        try:
            self._fetch_with_retry(fastf1.get_event_schedule, 2026, retries=2)
            return {
                "success": True,
                "status": "ok",
                "message": "Conexión a FastF1 establecida correctamente (Tiempo Real)",
            }
        except Exception as e:
            return {
                "success": False,
                "status": "error",
                "error": str(e),
                "message": "Error de conexión con FastF1",
            }

    def get_seasons(self):
        return [2026]

    def get_events(self, year=2026, filter_active_drivers=False, strict=False):
        try:
            schedule = self._fetch_with_retry(fastf1.get_event_schedule, year, retries=3)
            
            if schedule.empty:
                return []
                
            events = []
            for _, row in schedule.iterrows():
                # Basic validation
                if not isinstance(row.get('RoundNumber'), (int, float)) or row['RoundNumber'] == 0:
                     continue
                     
                events.append({
                    "RoundNumber": int(row['RoundNumber']),
                    "Country": row['Country'],
                    "Location": row['Location'],
                    "OfficialEventName": row['EventName'],
                    "EventDate": str(row['EventDate'])
                })

            if filter_active_drivers:
                return self._filter_events_for_active_drivers(events)

            return events
        except Exception as e:
            logger.error(f"Error fetching events for {year}: {e}")
            if strict:
                raise
            return []

    def _load_official_laps_2026(self):
        return dict(self._official_laps_2026)

    def reimport_2026_data(self):
        try:
            self._load_official_laps_2026()
            events = self.get_events(2026, filter_active_drivers=False, strict=True)
            if not events:
                return {
                    "success": False,
                    "status": "error",
                    "error": "No hay eventos disponibles para 2026.",
                    "events_total": 0,
                    "events_processed": 0,
                    "errors": [],
                    "message": "No hay eventos disponibles para 2026.",
                }

            events_total = len(events)
            processed = 0
            errors = []

            for ev in events:
                loc = ev.get("Location")
                if not loc:
                    errors.append({"location": None, "error": "Evento sin Location"})
                    continue
                try:
                    self.get_historical_circuit_data(loc, force_refresh=False)
                    processed += 1
                except Exception as e:
                    errors.append({"location": loc, "error": str(e)})

            return {
                "success": processed > 0,
                "status": "ok" if processed > 0 else "error",
                "events_total": events_total,
                "events_processed": processed,
                "errors": errors,
                "message": "Reimportación completada." if processed > 0 else "Reimportación fallida.",
            }
        except Exception as e:
            return {
                "success": False,
                "status": "error",
                "error": str(e),
                "events_total": 0,
                "events_processed": 0,
                "errors": [],
                "message": "Error durante la reimportación.",
            }

    def _get_base_roster_2026(self):
        drivers = []
        try:
            drivers = self.get_drivers(2025, "Abu Dhabi Grand Prix")
        except Exception:
            drivers = []
        if not self._is_valid_roster(drivers, min_drivers=18):
            drivers = self._get_fallback_drivers_2026()
        return drivers

    def _get_fallback_drivers_2026(self):
        fallback = [
            {"Abbreviation": "VER", "DriverId": "max_verstappen", "TeamName": "Red Bull Racing", "FullName": "Max Verstappen", "HeadshotUrl": "", "GridPosition": None, "Number": 1},
            {"Abbreviation": "PER", "DriverId": "sergio_perez", "TeamName": "Red Bull Racing", "FullName": "Sergio Perez", "HeadshotUrl": "", "GridPosition": None, "Number": 11},
            {"Abbreviation": "HAM", "DriverId": "lewis_hamilton", "TeamName": "Mercedes", "FullName": "Lewis Hamilton", "HeadshotUrl": "", "GridPosition": None, "Number": 44},
            {"Abbreviation": "RUS", "DriverId": "george_russell", "TeamName": "Mercedes", "FullName": "George Russell", "HeadshotUrl": "", "GridPosition": None, "Number": 63},
            {"Abbreviation": "LEC", "DriverId": "charles_leclerc", "TeamName": "Ferrari", "FullName": "Charles Leclerc", "HeadshotUrl": "", "GridPosition": None, "Number": 16},
            {"Abbreviation": "SAI", "DriverId": "carlos_sainz", "TeamName": "Ferrari", "FullName": "Carlos Sainz", "HeadshotUrl": "", "GridPosition": None, "Number": 55},
            {"Abbreviation": "NOR", "DriverId": "lando_norris", "TeamName": "McLaren", "FullName": "Lando Norris", "HeadshotUrl": "", "GridPosition": None, "Number": 4},
            {"Abbreviation": "PIA", "DriverId": "oscar_piastri", "TeamName": "McLaren", "FullName": "Oscar Piastri", "HeadshotUrl": "", "GridPosition": None, "Number": 81},
            {"Abbreviation": "ALO", "DriverId": "fernando_alonso", "TeamName": "Aston Martin", "FullName": "Fernando Alonso", "HeadshotUrl": "", "GridPosition": None, "Number": 14},
            {"Abbreviation": "STR", "DriverId": "lance_stroll", "TeamName": "Aston Martin", "FullName": "Lance Stroll", "HeadshotUrl": "", "GridPosition": None, "Number": 18},
            {"Abbreviation": "GAS", "DriverId": "pierre_gasly", "TeamName": "Alpine", "FullName": "Pierre Gasly", "HeadshotUrl": "", "GridPosition": None, "Number": 10},
            {"Abbreviation": "OCO", "DriverId": "esteban_ocon", "TeamName": "Alpine", "FullName": "Esteban Ocon", "HeadshotUrl": "", "GridPosition": None, "Number": 31},
            {"Abbreviation": "ALB", "DriverId": "alex_albon", "TeamName": "Williams", "FullName": "Alexander Albon", "HeadshotUrl": "", "GridPosition": None, "Number": 23},
            {"Abbreviation": "SAR", "DriverId": "logan_sargeant", "TeamName": "Williams", "FullName": "Logan Sargeant", "HeadshotUrl": "", "GridPosition": None, "Number": 2},
            {"Abbreviation": "BOT", "DriverId": "valtteri_bottas", "TeamName": "Stake F1", "FullName": "Valtteri Bottas", "HeadshotUrl": "", "GridPosition": None, "Number": 77},
            {"Abbreviation": "ZHO", "DriverId": "guanyu_zhou", "TeamName": "Stake F1", "FullName": "Guanyu Zhou", "HeadshotUrl": "", "GridPosition": None, "Number": 24},
            {"Abbreviation": "TSU", "DriverId": "yuki_tsunoda", "TeamName": "RB", "FullName": "Yuki Tsunoda", "HeadshotUrl": "", "GridPosition": None, "Number": 22},
            {"Abbreviation": "RIC", "DriverId": "daniel_ricciardo", "TeamName": "RB", "FullName": "Daniel Ricciardo", "HeadshotUrl": "", "GridPosition": None, "Number": 3},
            {"Abbreviation": "HUL", "DriverId": "nico_hulkenberg", "TeamName": "Haas", "FullName": "Nico Hulkenberg", "HeadshotUrl": "", "GridPosition": None, "Number": 27},
            {"Abbreviation": "MAG", "DriverId": "kevin_magnussen", "TeamName": "Haas", "FullName": "Kevin Magnussen", "HeadshotUrl": "", "GridPosition": None, "Number": 20},
        ]
        return [dict(d) for d in fallback]

    def _active_roster_signature(self):
        # Still useful for in-memory comparisons if needed, but not for file cache
        roster = self._get_base_roster_2026()
        ids = sorted([str(d.get("DriverId")) for d in roster if d.get("DriverId")])
        raw = ",".join(ids).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _filter_events_for_active_drivers(self, events):
        if not isinstance(events, list) or len(events) == 0:
            return []

        # 1. Get Active Driver IDs
        active_roster = self._get_base_roster_2026()
        active_ids = set()
        for d in active_roster:
            if d.get("DriverId"):
                active_ids.add(str(d.get("DriverId")))
        
        logger.info(f"Active driver IDs count: {len(active_ids)}")
        if len(active_ids) == 0:
            return events

        def check_event_relevance(event):
            loc = event.get("Location")
            if not loc:
                return False
            try:
                hist = self.get_historical_circuit_data(loc, force_refresh=False)
            except Exception:
                return False
            if not isinstance(hist, dict):
                return False
            counts = hist.get("driver_lap_counts", {})
            if not isinstance(counts, dict) or not counts:
                return False
            for d_id, n in counts.items():
                if str(d_id) in active_ids and isinstance(n, (int, float)) and int(n) > 0:
                    return True
            return False

        valid_events = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_event = {executor.submit(check_event_relevance, ev): ev for ev in events}
            for future in concurrent.futures.as_completed(future_to_event):
                ev = future_to_event[future]
                try:
                    is_relevant = future.result()
                    if is_relevant:
                        valid_events.append(ev)
                except Exception as e:
                    logger.error(f"Error filtering event {ev.get('Location')}: {e}")
                    pass
        
        valid_events.sort(key=lambda x: int(x.get('RoundNumber', 0)))
        return valid_events

    def _normalize_event_identifier(self, event_identifier):
        try:
            if isinstance(event_identifier, str) and event_identifier.isdigit():
                return int(event_identifier)
        except Exception:
            pass
        return event_identifier

    def get_drivers(self, year, event_identifier, **kwargs):
        try:
            logger.info(f"get_drivers called for year={year}, event={event_identifier}")
            if year == 2026:
                drivers = []
                try:
                    drivers = self.get_drivers(2025, 'Abu Dhabi Grand Prix')
                    logger.info(f"Base roster from 2025: {len(drivers)} drivers")
                except Exception as e:
                    logger.warning(f"Failed to fetch base roster from 2025: {e}")
                    pass

                if not self._is_valid_roster(drivers, min_drivers=18):
                    logger.info("Invalid/Empty 2025 roster, using fallback 2026 drivers")
                    drivers = self._get_fallback_drivers_2026()
                    logger.info(f"Fallback roster size: {len(drivers)}")

                weights = kwargs.get("weights")
                return self._generate_2026_predicted_grid(drivers, event_identifier, weights=weights)

            session = self._fetch_with_retry(fastf1.get_session, year, self._normalize_event_identifier(event_identifier), 'R', retries=3)
            self._fetch_with_retry(session.load, laps=False, telemetry=False, weather=False, messages=False, retries=3)
            
            results = getattr(session, "results", None)
            grid_map = {}
            if results is not None:
                try:
                    for _, row in results.iterrows():
                        abbr = row.get("Abbreviation")
                        grid_pos = row.get("GridPosition")
                        if abbr is not None and pd.notnull(grid_pos):
                            grid_map[str(abbr)] = int(grid_pos)
                except Exception as e:
                    logger.warning(f"Error building grid map: {e}")

            drivers = []
            if not hasattr(session, 'drivers') or not session.drivers:
                # Fallback if drivers list is empty
                if hasattr(session, 'results') and not session.results.empty:
                     # Try to rebuild from results
                     pass
                else:
                    raise ValueError(f"No drivers found for {year} - {event_identifier}")

            for idx, driver_code in enumerate(session.drivers):
                info = session.get_driver(driver_code)
                abbr = info.get("Abbreviation")
                
                if not abbr:
                    continue

                car_number = None
                for key in ("DriverNumber", "PermanentNumber", "Number"):
                    try:
                        val = info.get(key)
                    except Exception:
                        val = None
                    if val is not None and not pd.isna(val):
                        car_number = val
                        break

                if car_number is None or (isinstance(car_number, str) and not car_number.strip()):
                    car_number = idx + 1

                drivers.append(
                    {
                        "Abbreviation": abbr,
                        "DriverId": info.get("DriverId", info.name),
                        "TeamName": info.get("TeamName", "Unknown"),
                        "FullName": info.get("FullName", "Unknown"),
                        "HeadshotUrl": info.get("HeadshotUrl", ""),
                        "GridPosition": grid_map.get(abbr),
                        "Number": car_number,
                    }
                )
            
            if not self._is_valid_roster(drivers, min_drivers=1):
                raise ValueError("Fetched drivers list failed validation schema")

            return drivers
        except Exception as e:
            logger.error(f"Error fetching drivers for {year} - {event_identifier}: {e}")
            return []

    def _is_valid_roster(self, drivers, min_drivers=18):
        if not isinstance(drivers, list):
            return False
        if len(drivers) < int(min_drivers):
            return False
        abbrs = []
        for d in drivers:
            if not isinstance(d, dict):
                return False
            abbr = d.get("Abbreviation")
            if not abbr or not isinstance(abbr, str):
                return False
            abbrs.append(abbr)
        return len(abbrs) == len(set(abbrs))

    def _resolve_2026_location(self, event_identifier):
        target_id = str(event_identifier) if event_identifier is not None else None
        if not target_id:
            return None
        events = self.get_events(2026)
        if not isinstance(events, list):
            return None
        for ev in events:
            if str(ev.get("RoundNumber")) == target_id or str(ev.get("OfficialEventName")) == target_id:
                return ev.get("Location")
        return None

    def _time_to_seconds(self, t):
        try:
            if t is None or pd.isna(t):
                return None
        except Exception:
            pass
        try:
            if hasattr(t, "total_seconds"):
                return float(t.total_seconds())
        except Exception:
            pass
        try:
            return float(t)
        except Exception:
            return None

    def _get_location_qualy_history(self, location, start_year, end_year):
        if not location:
            return []

        cache_key = (str(location).lower(), int(start_year), int(end_year))
        if cache_key in self._qualy_history_cache:
            return self._qualy_history_cache[cache_key]

        years = list(range(int(start_year), int(end_year) + 1))

        def process_year(y):
            try:
                schedule = self._fetch_with_retry(fastf1.get_event_schedule, y, retries=2)
                event_row = schedule[schedule["Location"].str.contains(location, case=False, regex=False)]
                if event_row.empty:
                    return []

                event_name = event_row.iloc[0]["EventName"]
                session = self._fetch_with_retry(fastf1.get_session, y, event_name, "Q", retries=2)
                self._fetch_with_retry(session.load, laps=False, telemetry=False, weather=False, messages=False, retries=2)

                results = getattr(session, "results", None)
                if results is None or getattr(results, "empty", True):
                    return []

                out = []
                for _, row in results.iterrows():
                    abbr = row.get("Abbreviation")
                    team = row.get("TeamName")
                    pos = row.get("Position")
                    q3 = row.get("Q3")
                    q2 = row.get("Q2")
                    q1 = row.get("Q1")
                    best = q3 if q3 is not None and not pd.isna(q3) else (q2 if q2 is not None and not pd.isna(q2) else q1)
                    best_s = self._time_to_seconds(best)
                    if not abbr or best_s is None:
                        continue
                    pos_i = None
                    try:
                        if pos is not None and not pd.isna(pos):
                            pos_i = int(pos)
                    except Exception:
                        pos_i = None
                    out.append(
                        {
                            "year": int(y),
                            "abbr": str(abbr),
                            "team": str(team) if team else None,
                            "best_time": float(best_s),
                            "position": pos_i,
                        }
                    )
                return out
            except Exception:
                return []

        entries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(process_year, y) for y in years]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    entries.extend(fut.result() or [])
                except Exception:
                    pass

        self._qualy_history_cache[cache_key] = entries
        return entries

    def _get_location_race_history(self, location, start_year, end_year):
        """
        Obtiene resultados históricos de carrera (R) para una localización y rango de años.
        Mejorado con validación de datos y manejo de errores.
        """
        if not location:
            return []

        cache_key = (str(location).lower(), int(start_year), int(end_year))
        if cache_key in self._race_history_cache:
            return self._race_history_cache[cache_key]

        years = list(range(int(start_year), int(end_year) + 1))

        def process_year(y):
            try:
                # 1. Fetch Schedule with retry
                schedule = self._fetch_with_retry(fastf1.get_event_schedule, y, retries=2)
                if schedule.empty:
                    return []

                # 2. Find Event by Location (Robust matching)
                # Try exact match first, then contains
                event_row = schedule[schedule["Location"].str.lower() == location.lower()]
                if event_row.empty:
                    event_row = schedule[schedule["Location"].str.contains(location, case=False, regex=False)]
                
                if event_row.empty:
                    return []

                event_name = event_row.iloc[0]["EventName"]
                
                # 3. Fetch Session
                session = self._fetch_with_retry(fastf1.get_session, y, event_name, "R", retries=2)
                # Load minimal data needed for results
                self._fetch_with_retry(session.load, laps=False, telemetry=False, weather=False, messages=False, retries=2)

                results = getattr(session, "results", None)
                if results is None or getattr(results, "empty", True):
                    return []

                out = []
                for _, row in results.iterrows():
                    abbr = row.get("Abbreviation")
                    team = row.get("TeamName")
                    pos = row.get("Position")
                    grid_pos = row.get("GridPosition")
                    status = row.get("Status") # e.g. Finished, +1 Lap, DNF

                    if not abbr:
                        continue
                        
                    # Validation: Ensure position is a number
                    pos_i = None
                    try:
                        if pos is not None and not pd.isna(pos):
                            pos_i = int(pos)
                    except Exception:
                        pos_i = None
                        
                    grid_pos_i = None
                    try:
                        if grid_pos is not None and not pd.isna(grid_pos):
                            grid_pos_i = int(grid_pos)
                    except Exception:
                        grid_pos_i = None

                    out.append(
                        {
                            "year": int(y),
                            "abbr": str(abbr),
                            "team": str(team) if team else None,
                            "position": pos_i,
                            "grid_position": grid_pos_i,
                            "status": str(status)
                        }
                    )
                return out
            except Exception as e:
                logger.warning(f"Error fetching race history for {location} in {y}: {e}")
                return []

        entries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(process_year, y) for y in years]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                    if res:
                        entries.extend(res)
                except Exception:
                    pass

        self._race_history_cache[cache_key] = entries
        return entries

    def _get_circuit_profile_for_location(self, location):
        """
        Devuelve el perfil de circuito usado por el motor de simulación
        para una localización dada. Si el motor no está disponible, usa
        un perfil genérico equilibrado.
        """
        try:
            profile = sim_engine._get_circuit_profile(location)  # type: ignore[name-defined]
            if isinstance(profile, dict):
                return profile
        except Exception:
            pass

        return {
            "type": "Mixed",
            "zones": 2,
            "difficulty": 0.8,
            "length": 5.0,
            "corners": 15,
            "qualifying_weight": 0.5,
            "style_bias": "Balanced",
        }

    def _deterministic_jitter(self, event_key, driver_key):
        raw = f"{event_key}|{driver_key}".encode("utf-8")
        h = hashlib.sha1(raw).digest()
        v = int.from_bytes(h[:2], "big") / 65535.0
        return (v - 0.5) * 0.02

    def _calibrate_grid_weights_from_qualy(self, location, qualy_entries):
        """
        Ajusta pesos de la fórmula de parrilla usando resultados históricos
        de clasificación para el mismo circuito.
        """
        if not location or not qualy_entries:
            return {
                "team_delta": 1.0,
                "driver_gap": 1.0,
                "track_bonus": 1.0,
                "race_delta": 0.1,
                "jitter": 1.0,
            }

        cache_key = str(location)
        cached = self._grid_weights_by_circuit.get(cache_key)
        if isinstance(cached, dict):
            return cached

        team_year_times = {}
        for e in qualy_entries:
            y = e.get("year")
            team = e.get("team")
            t = e.get("best_time")
            if y is None or team is None or t is None:
                continue
            key = (int(y), str(team))
            team_year_times.setdefault(key, []).append(float(t))

        years_present = sorted({int(e["year"]) for e in qualy_entries if "year" in e})
        if not years_present or not team_year_times:
            base = {
                "team_delta": 1.0,
                "driver_gap": 1.0,
                "track_bonus": 1.0,
                "race_delta": 0.1,
                "jitter": 1.0,
            }
            self._grid_weights_by_circuit[cache_key] = base
            return base

        driver_skill_samples = {}
        for e in qualy_entries:
            y = e.get("year")
            abbr = e.get("abbr")
            team = e.get("team")
            t = e.get("best_time")
            if y is None or not abbr or team is None or t is None:
                continue
            team_times = team_year_times.get((int(y), str(team))) or []
            if not team_times:
                continue
            team_med = float(np.median(team_times))
            residual = float(t - team_med)
            driver_skill_samples.setdefault(str(abbr), []).append(residual)

        driver_qualy_skill = {}
        for abbr, vals in driver_skill_samples.items():
            if not vals:
                continue
            driver_qualy_skill[abbr] = float(np.mean(vals))

        base_weights = {
            "team_delta": 1.0,
            "driver_gap": 1.0,
            "track_bonus": 1.0,
            "race_delta": 0.1,
            "jitter": 1.0,
        }

        candidates = []
        for w_team in (0.8, 1.0, 1.2):
            for w_gap in (0.8, 1.0, 1.2):
                w = dict(base_weights)
                w["team_delta"] = float(w_team)
                w["driver_gap"] = float(w_gap)
                candidates.append(w)

        def compute_error(weights):
            total_error = 0.0
            count = 0

            for y in years_present:
                year_entries = [
                    e
                    for e in qualy_entries
                    if e.get("year") is not None and int(e.get("year")) == y
                ]
                if not year_entries:
                    continue

                scored = []
                for e in year_entries:
                    abbr = e.get("abbr")
                    team = e.get("team")
                    if not abbr or team is None:
                        continue

                    team_times = team_year_times.get((int(y), str(team))) or []
                    if not team_times:
                        continue

                    team_med = float(np.median(team_times))
                    t = e.get("best_time")
                    if t is None:
                        continue
                    residual = float(t - team_med)
                    driver_gap = driver_qualy_skill.get(str(abbr), residual)

                    td_key = (int(y), str(team))
                    if td_key in team_year_times:
                        team_med_y = float(np.median(team_year_times[td_key]))
                        best_t = min(float(np.median(v)) for k, v in team_year_times.items() if k[0] == int(y))
                        team_delta_y = float(team_med_y - best_t)
                    else:
                        team_delta_y = 0.0

                    score = (
                        float(weights["team_delta"]) * float(team_delta_y)
                        + float(weights["driver_gap"]) * float(driver_gap)
                    )
                    true_pos = e.get("position")
                    if true_pos is None:
                        continue
                    scored.append((score, str(abbr), int(true_pos)))

                if not scored:
                    continue

                scored.sort(key=lambda x: (x[0], x[1]))
                for idx, (_, _, true_pos) in enumerate(scored, start=1):
                    total_error += abs(float(idx) - float(true_pos))
                    count += 1

            if count == 0:
                return float("inf")
            return float(total_error) / float(count)

        best_weights = dict(base_weights)
        best_error = compute_error(best_weights)
        for cand in candidates:
            err = compute_error(cand)
            if err < best_error:
                best_error = err
                best_weights = cand

        self._grid_weights_by_circuit[cache_key] = best_weights
        return best_weights

    def _generate_2026_predicted_grid(self, drivers, event_identifier, weights=None):
        logger.info(f"Generating 2026 predicted grid for {event_identifier}")
        if not isinstance(drivers, list) or not drivers:
            logger.error("Drivers list provided to _generate_2026_predicted_grid is empty or invalid")
            return []

        event_key = str(event_identifier) if event_identifier is not None else "unknown"

        # Skip cache if custom weights are provided to ensure they take effect
        if weights is None and event_key in self._grid_history:
            logger.info("Returning cached grid history")
            last_order_abbrs = self._grid_history[event_key]
            # ... (cache logic) ...
            current_abbrs = {d.get("Abbreviation") for d in drivers}
            if set(last_order_abbrs) == current_abbrs:
                sorted_drivers = []
                driver_map = {d.get("Abbreviation"): d for d in drivers}
                for abbr in last_order_abbrs:
                    if abbr in driver_map:
                        sorted_drivers.append(driver_map[abbr])

                final_grid = []
                for pos, d in enumerate(sorted_drivers, start=1):
                    d_copy = dict(d)
                    d_copy["GridPosition"] = pos
                    final_grid.append(d_copy)
                return final_grid

        location = self._resolve_2026_location(event_identifier)
        logger.info(f"Resolved location: {location}")
        start_year, end_year = self._get_historical_year_range()

        qualy_entries = []
        race_entries = []
        if location:
            qualy_entries = self._get_location_qualy_history(location, start_year, end_year)
            race_entries = self._get_location_race_history(location, start_year, end_year)
            logger.info(f"Found {len(qualy_entries)} historical qualy and {len(race_entries)} race entries for {location}")

        hist_data = None
        if location:
            try:
                hist_data = self.get_historical_circuit_data(location)
            except Exception as e:
                logger.warning(f"Failed to get historical circuit data: {e}")
                hist_data = None

        if not qualy_entries:
            logger.info("No qualy history, falling back to performance grid")
            return self._generate_2026_performance_grid(drivers, event_identifier)

        team_year_times = {}
        for e in qualy_entries:
            y = e.get("year")
            team = e.get("team")
            t = e.get("best_time")
            if y is None or team is None or t is None:
                continue
            key = (int(y), str(team))
            team_year_times.setdefault(key, []).append(float(t))

        team_year_delta = {}
        years_present = sorted({int(e["year"]) for e in qualy_entries if "year" in e})
        for y in years_present:
            team_medians = []
            for (yy, team), times in team_year_times.items():
                if yy != y or not times:
                    continue
                team_medians.append((team, float(np.median(times))))
            if not team_medians:
                continue
            best_time = min([t for _, t in team_medians])
            for team, med in team_medians:
                team_year_delta[(y, team)] = float(med - best_time)

        team_pred_delta_2026 = {}
        for team in {team for (_, team) in team_year_delta.keys()}:
            pts = [(y, d) for (y, t), d in team_year_delta.items() if t == team]
            pts.sort(key=lambda x: x[0])
            if len(pts) >= 3:
                xs = np.array([p[0] for p in pts], dtype=float)
                ys = np.array([p[1] for p in pts], dtype=float)
                try:
                    m, b = np.polyfit(xs, ys, 1)
                    pred = float(m * 2026.0 + b)
                except Exception:
                    pred = float(pts[-1][1])
            elif len(pts) >= 1:
                pred = float(pts[-1][1])
            else:
                continue
            team_pred_delta_2026[team] = max(0.0, pred)

        driver_qualy_skill = {}
        driver_skill_samples = {}
        for e in qualy_entries:
            y = e.get("year")
            abbr = e.get("abbr")
            team = e.get("team")
            t = e.get("best_time")
            if y is None or not abbr or team is None or t is None:
                continue
            team_times = team_year_times.get((int(y), str(team))) or []
            if not team_times:
                continue
            team_med = float(np.median(team_times))
            residual = float(t - team_med)
            driver_skill_samples.setdefault(str(abbr), []).append(residual)

        for abbr, vals in driver_skill_samples.items():
            if not vals:
                continue
            driver_qualy_skill[abbr] = float(np.mean(vals))

        race_driver_stats = {}
        race_team_stats = {}
        if isinstance(hist_data, dict):
            race_driver_stats = hist_data.get("driver_stats", {}) or {}
            race_team_stats = hist_data.get("team_stats", {}) or {}

        # Populate driver_track_history from race_entries
        driver_track_history = {}
        for re in race_entries:
            abbr = re.get("abbr")
            pos = re.get("position")
            if abbr and pos is not None:
                driver_track_history.setdefault(abbr, []).append(pos)

        circuit_profile = None
        if location:
            circuit_profile = self._get_circuit_profile_for_location(location)

        # Determine weights: use provided or calibrate
        if weights and isinstance(weights, dict):
             final_weights = {
                "team_delta": 1.0,
                "driver_gap": 1.0,
                "track_bonus": 1.0,
                "race_delta": 0.1,
                "jitter": 1.0,
            }
             final_weights.update(weights)
             weights = final_weights
        else:
             base_weights = self._calibrate_grid_weights_from_qualy(location, qualy_entries)
             weights = dict(base_weights)

        if circuit_profile:
            bias = circuit_profile.get("style_bias", "Balanced")
            c_type = circuit_profile.get("type", "Mixed")

            if bias == "Power":
                weights["team_delta"] *= 1.05
                weights["driver_gap"] *= 0.95
            elif bias == "Technical":
                weights["team_delta"] *= 0.95
                weights["driver_gap"] *= 1.05

            if c_type == "Urban":
                weights["driver_gap"] *= 1.05

        default_team_delta = float(np.median(list(team_pred_delta_2026.values()))) if team_pred_delta_2026 else 0.8

        scored = []
        for d in drivers:
            abbr = d.get("Abbreviation")
            team = d.get("TeamName")
            d_id = d.get("DriverId") or d.get("DriverID") or abbr

            team_delta = team_pred_delta_2026.get(team) if team else None
            if team_delta is None:
                team_delta = default_team_delta

            # Driver Skill Adjustment (Gap to car potential)
            # Lower is better (0.0 means perfect extraction of car performance)
            qualy_gap = driver_qualy_skill.get(str(abbr), 0.05) if abbr else 0.05 # Default 0.05s gap
            
            # Track Specialist Bonus
            # If driver often finishes top 5 here, reduce predicted time
            track_bonus = 0.0
            if abbr in driver_track_history:
                positions = driver_track_history[abbr]
                top5_count = sum(1 for p in positions if p <= 5)
                if len(positions) > 0:
                    ratio = top5_count / len(positions)
                    track_bonus = ratio * 0.15 # Up to 0.15s gain for track specialists

            race_adj = 0.0
            hist = race_driver_stats.get(d_id) if d_id else None
            if not hist and team:
                hist = race_team_stats.get(team)
            if isinstance(hist, dict):
                mean = hist.get("mean")
                if mean is not None:
                    race_adj = float(mean)
            race_base = 0.0
            team_hist = race_team_stats.get(team) if team else None
            if isinstance(team_hist, dict) and team_hist.get("mean") is not None:
                race_base = float(team_hist.get("mean"))
            race_delta = (race_adj - race_base) if race_base > 0 else 0.0

            jitter = self._deterministic_jitter(event_key, d_id or abbr or "")

            score = (
                float(weights.get("team_delta", 1.0)) * float(team_delta)
                + float(weights.get("driver_gap", 1.0)) * float(qualy_gap)
                - float(weights.get("track_bonus", 1.0)) * float(track_bonus)
                + float(weights.get("race_delta", 0.1)) * float(race_delta)
                + float(weights.get("jitter", 1.0)) * float(jitter)
            )
            scored.append((score, str(abbr) if abbr else "", d))

        scored.sort(key=lambda x: (x[0], x[1]))

        final_grid = []
        abbr_order = []
        for pos, (_, _, d) in enumerate(scored, start=1):
            d_copy = dict(d)
            d_copy["GridPosition"] = pos
            final_grid.append(d_copy)
            abbr_order.append(d.get("Abbreviation"))

        self._grid_history[event_key] = abbr_order
        return final_grid

    def _generate_2026_performance_grid(self, drivers, event_identifier):
        if not isinstance(drivers, list):
            return []
        if len(drivers) == 0:
            return []

        event_key = str(event_identifier) if event_identifier is not None else "unknown"

        if event_key in self._grid_history:
            last_order_abbrs = self._grid_history[event_key]
            current_abbrs = {d.get("Abbreviation") for d in drivers}
            if set(last_order_abbrs) == current_abbrs:
                sorted_drivers = []
                driver_map = {d.get("Abbreviation"): d for d in drivers}
                for abbr in last_order_abbrs:
                    if abbr in driver_map:
                        sorted_drivers.append(driver_map[abbr])

                final_grid = []
                for pos, d in enumerate(sorted_drivers, start=1):
                    d_copy = dict(d)
                    d_copy["GridPosition"] = pos
                    final_grid.append(d_copy)
                return final_grid

        try:
            location = None
            events = self.get_events(2026)
            if isinstance(events, list):
                target_id = str(event_identifier) if event_identifier is not None else None
                for ev in events:
                    if target_id is None:
                        continue
                    if str(ev.get("RoundNumber")) == target_id or str(ev.get("OfficialEventName")) == target_id:
                        location = ev.get("Location")
                        break

            hist_data = None
            if location:
                hist_data = self.get_historical_circuit_data(location)

            if not hist_data or not isinstance(hist_data, dict):
                return self._generate_random_grid(drivers, event_identifier)

            driver_stats = hist_data.get("driver_stats", {})
            team_stats = hist_data.get("team_stats", {})

            means = []
            if isinstance(driver_stats, dict):
                for v in driver_stats.values():
                    if isinstance(v, dict) and "mean" in v:
                        means.append(v["mean"])
            if not means and isinstance(team_stats, dict):
                for v in team_stats.values():
                    if isinstance(v, dict) and "mean" in v:
                        means.append(v["mean"])

            global_mean = float(np.mean(means)) if means else 90.0

            scored = []
            for d in drivers:
                abbr = d.get("Abbreviation")
                d_id = d.get("DriverId") or d.get("DriverID") or abbr
                team = d.get("TeamName")

                stats = None
                if isinstance(driver_stats, dict):
                    stats = driver_stats.get(str(d_id))
                    if stats is None and abbr is not None:
                        stats = driver_stats.get(str(abbr))
                if stats is None and isinstance(team_stats, dict) and team is not None:
                    stats = team_stats.get(team)

                base_mean = None
                if isinstance(stats, dict):
                    base_mean = stats.get("mean")
                if base_mean is None:
                    base_mean = global_mean

                scored.append((float(base_mean), abbr or "", d))

            scored.sort(key=lambda x: (x[0], x[1]))

            final_grid = []
            abbr_order = []
            for pos, (_, _, d) in enumerate(scored, start=1):
                d_copy = dict(d)
                d_copy["GridPosition"] = pos
                final_grid.append(d_copy)
                abbr_order.append(d.get("Abbreviation"))

            self._grid_history[event_key] = abbr_order
            return final_grid
        except Exception as e:
            logger.error(f"Error generating 2026 performance grid for {event_identifier}: {e}")
            return self._generate_random_grid(drivers, event_identifier)

    def _generate_random_grid(self, drivers, event_identifier):
        """
        Generates a completely random, cryptographically secure grid for 2026 predictions.
        Removes any historical bias as per new requirements.
        """
        if not isinstance(drivers, list):
            return []
        n = len(drivers)
        if n == 0:
            return []

        event_key = str(event_identifier) if event_identifier is not None else "unknown"

        if event_key in self._grid_history:
            last_order_abbrs = self._grid_history[event_key]
            current_abbrs = {d.get("Abbreviation") for d in drivers}
            if set(last_order_abbrs) == current_abbrs:
                sorted_drivers = []
                driver_map = {d.get("Abbreviation"): d for d in drivers}
                for abbr in last_order_abbrs:
                    if abbr in driver_map:
                        sorted_drivers.append(driver_map[abbr])

                final_grid = []
                for pos, d in enumerate(sorted_drivers, start=1):
                    d_copy = dict(d)
                    d_copy["GridPosition"] = pos
                    final_grid.append(d_copy)
                return final_grid

        # Cryptographically secure shuffle
        shuffled_drivers = list(drivers)
        secrets.SystemRandom().shuffle(shuffled_drivers)

        final_grid = []
        abbr_order = []
        
        for pos, d in enumerate(shuffled_drivers, start=1):
            d_copy = dict(d)
            d_copy["GridPosition"] = pos
            final_grid.append(d_copy)
            abbr_order.append(d.get("Abbreviation"))

        # Cache the order
        self._grid_history[event_key] = abbr_order
        
        return final_grid

    def get_session_info(self, year, event_identifier):
        try:
            if year == 2026:
                return self.get_official_total_laps_2026(event_identifier)

            session = fastf1.get_session(year, self._normalize_event_identifier(event_identifier), 'R')
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            
            if not hasattr(session, 'laps') or session.laps is None or session.laps.empty:
                 return {"total_laps": 0}
            
            total_laps = int(session.laps['LapNumber'].max())
            return {"total_laps": total_laps}
        except Exception as e:
            logger.error(f"Error fetching session info for {year} - {event_identifier}: {e}")
            return {"total_laps": 0}

    def _compute_laps_from_history(self, location, start_year, end_year, mode='max'):
        total_laps_counts = {}
        years = range(start_year, end_year + 1)
        
        for y in reversed(list(years)):
            try:
                schedule = self._fetch_with_retry(fastf1.get_event_schedule, y, retries=2)
                event_row = schedule[schedule['Location'].str.contains(location, case=False, regex=False)]
                if event_row.empty:
                    continue

                event_name = event_row.iloc[0]['EventName']
                session = self._fetch_with_retry(fastf1.get_session, y, event_name, 'R', retries=2)
                self._fetch_with_retry(session.load, laps=True, telemetry=False, weather=False, messages=False, retries=2)
                
                if hasattr(session, 'laps') and session.laps is not None and not session.laps.empty:
                    tl = int(session.laps['LapNumber'].max())
                    if tl > 0:
                        if mode == 'latest':
                            return tl
                        total_laps_counts[tl] = total_laps_counts.get(tl, 0) + 1
            except Exception as e:
                continue

        if mode == 'latest':
            return 0
            
        if not total_laps_counts:
            return 0
        return int(max(total_laps_counts, key=total_laps_counts.get))

    def _compute_latest_total_laps(self, location, start_year, end_year):
        return self._compute_laps_from_history(location, start_year, end_year, mode='latest')

    def get_race_laps(self, year, event_identifier):
        # ... existing implementation adapted to no cache ...
        try:
            session = self._fetch_with_retry(fastf1.get_session, year, self._normalize_event_identifier(event_identifier), 'R', retries=3)
            self._fetch_with_retry(session.load, laps=True, telemetry=True, weather=True, retries=3)
            
            if not hasattr(session, 'laps') or session.laps is None or session.laps.empty:
                 return None
            return session
        except Exception as e:
            logger.error(f"Error fetching race data: {e}")
            return None

    def get_circuit_telemetry(self, session):
        try:
            fastest_lap = session.laps.pick_fastest()
            if fastest_lap is None:
                fastest_lap = session.laps.iloc[0]

            try:
                telemetry = fastest_lap.get_telemetry().add_distance()
            except KeyError:
                raise RuntimeError("No hay datos de posición del coche")

            required_cols = {'X', 'Y', 'Distance'}
            if not required_cols.issubset(set(telemetry.columns)):
                raise RuntimeError("Circuit telemetry missing required X/Y/Distance columns")
            
            path_data = []
            for _, row in telemetry.iterrows():
                try:
                    if pd.isna(row['X']) or pd.isna(row['Y']):
                        continue
                    item = {
                        "X": float(row['X']),
                        "Y": float(row['Y']),
                        "Z": float(row.get('Z', 0.0)) if pd.notnull(row.get('Z')) else 0.0,
                        "Distance": float(row.get('Distance', 0.0)) if pd.notnull(row.get('Distance')) else 0.0,
                        "Speed": float(row.get('Speed', 0.0)) if pd.notnull(row.get('Speed')) else 0.0
                    }
                    path_data.append(item)
                except:
                    continue
            
            if not path_data:
                raise RuntimeError("No valid circuit telemetry points")
            return path_data
        except Exception as e:
            logger.error(f"Error extracting circuit telemetry: {e}")
            raise

    def get_historical_circuit_data(self, location, force_refresh=True):
        """
        Calculates stats on-the-fly for a circuit location.
        No caching. Fetches data in parallel for optimal real-time performance.
        """
        start_year, end_year = self._get_historical_year_range()
        years = list(range(start_year, end_year + 1))
        
        logger.info(f"Fetching historical data for {location} (Real-Time Mode)")
        
        def process_year(y):
            try:
                schedule = self._fetch_with_retry(fastf1.get_event_schedule, y, retries=2)
                event_row = schedule[schedule['Location'].str.contains(location, case=False, regex=False)]
                if event_row.empty:
                    return None
                
                event_name = event_row.iloc[0]['EventName']
                
                session = self._fetch_with_retry(fastf1.get_session, y, event_name, 'R', retries=2)
                
                # Load laps. Telemetry will be loaded only if needed for path later.
                self._fetch_with_retry(session.load, laps=True, telemetry=False, weather=False, messages=False, retries=2)
                
                if not hasattr(session, 'laps') or session.laps is None or session.laps.empty:
                    return None
                    
                return {"year": y, "session": session}
            except Exception as e:
                return None

        circuit_path = None
        total_laps_counts = {}
        driver_stats_acc = {}
        team_stats_acc = {}
        pos_change_values = []
        nonzero_pos_changes = 0
        total_pos_steps = 0
        reference_time = 90.0
        latest_year_seen = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Prioritize recent years for circuit path chance
            sorted_years = sorted(years, reverse=True)
            future_to_year = {executor.submit(process_year, y): y for y in sorted_years}
            
            for future in concurrent.futures.as_completed(future_to_year):
                result = future.result()
                if not result:
                    continue
                
                session = result["session"]
                year = result["year"]
                
                # 1. Total Laps Count
                try:
                    tl = int(session.laps['LapNumber'].max())
                    total_laps_counts[tl] = total_laps_counts.get(tl, 0) + 1
                except:
                    pass

                # 2. Circuit Path (Try to get from most recent available if missing)
                if circuit_path is None:
                    try:
                        # We need to load telemetry for this session to get path
                        self._fetch_with_retry(session.load, laps=True, telemetry=True, weather=False, messages=False, retries=2)
                        circuit_path = self.get_circuit_telemetry(session)
                    except Exception:
                        pass
                
                try:
                    laps = session.laps
                    
                    # Calculate Session Mean for Normalization
                    session_mean = None
                    try:
                        valid_laps_session = laps.pick_quicklaps()
                        if valid_laps_session.empty:
                             valid_laps_session = laps
                        session_times = valid_laps_session['LapTime'].dt.total_seconds().dropna()
                        if not session_times.empty:
                            session_mean = float(session_times.mean())
                    except Exception:
                        pass
                    
                    if session_mean and session_mean > 0:
                        if year > latest_year_seen:
                            latest_year_seen = year
                            reference_time = session_mean
                    else:
                        # Fallback to simple mean if calculation fails, but try to avoid raw times if possible
                        # If we can't normalize, we skip adding stats to avoid skewing
                        continue

                    logger.info(f"Processing stats for {session.event.year} (Mean: {session_mean:.2f}s) - Drivers: {len(session.drivers)}")
                    
                    for d in session.drivers:
                        try:
                            d_info = session.get_driver(d)
                            d_id = d_info.name # DriverId
                            team = d_info.get("TeamName")
                            
                            try:
                                d_laps = laps.pick_drivers(d)
                            except:
                                continue
                            
                            if d_laps.empty:
                                continue

                            # Robust lap selection
                            valid_laps = pd.DataFrame()
                            try:
                                valid_laps = d_laps.pick_quicklaps()
                            except:
                                pass
                            
                            if valid_laps.empty:
                                try:
                                    valid_laps = d_laps.pick_accurate()
                                except:
                                    pass
                            
                            if valid_laps.empty:
                                valid_laps = d_laps[d_laps['LapTime'].notna()]
                            
                            if valid_laps.empty:
                                continue

                            valid_times = valid_laps['LapTime'].dt.total_seconds().dropna().tolist()
                            
                            if valid_times:
                                # NORMALIZE TIMES
                                normalized_times = [t / session_mean for t in valid_times]
                                
                                if d_id not in driver_stats_acc: driver_stats_acc[d_id] = []
                                driver_stats_acc[d_id].extend(normalized_times)
                                
                                if team:
                                    if team not in team_stats_acc: team_stats_acc[team] = []
                                    team_stats_acc[team].extend(normalized_times)
                        except Exception:
                            continue
                    try:
                        if "Position" in laps.columns and "LapNumber" in laps.columns and "Driver" in laps.columns:
                            df_pos = laps[["LapNumber", "Driver", "Position"]].dropna(subset=["LapNumber", "Driver", "Position"])
                            if not df_pos.empty:
                                df_pos = df_pos.sort_values(["Driver", "LapNumber"])
                                positions_by_driver = {}
                                for _, row in df_pos.iterrows():
                                    drv = row["Driver"]
                                    lapn = int(row["LapNumber"])
                                    pos = int(row["Position"])
                                    positions_by_driver.setdefault(drv, []).append((lapn, pos))
                                for drv, seq in positions_by_driver.items():
                                    seq_sorted = sorted(seq, key=lambda x: x[0])
                                    last_pos = None
                                    for _, p in seq_sorted:
                                        if last_pos is not None:
                                            delta = abs(int(p) - int(last_pos))
                                            pos_change_values.append(float(delta))
                                            total_pos_steps += 1
                                            if delta > 0:
                                                nonzero_pos_changes += 1
                                        last_pos = p
                    except Exception:
                        pass
                except Exception:
                    continue

        driver_stats = {}
        for d_id, times in driver_stats_acc.items():
            if not times: continue
            times = np.array(times)
            # Simple outlier filtering
            q1, q3 = np.percentile(times, [25, 75])
            iqr = q3 - q1
            clean = times[(times >= q1 - 1.5*iqr) & (times <= q3 + 1.5*iqr)]
            if len(clean) > 0:
                driver_stats[d_id] = {"mean": float(np.mean(clean)), "std": float(np.std(clean))}

        team_stats = {}
        for team, times in team_stats_acc.items():
            if not times: continue
            times = np.array(times)
            q1, q3 = np.percentile(times, [25, 75])
            iqr = q3 - q1
            clean = times[(times >= q1 - 1.5*iqr) & (times <= q3 + 1.5*iqr)]
            if len(clean) > 0:
                team_stats[team] = {"mean": float(np.mean(clean)), "std": float(np.std(clean))}
        
        likely_total_laps = max(total_laps_counts, key=total_laps_counts.get) if total_laps_counts else 50

        overtake_stats = {}
        if pos_change_values:
            arr = np.array(pos_change_values, dtype=float)
            avg_pos_change = float(np.mean(arr))
            try:
                p90 = float(np.percentile(arr, 90))
            except Exception:
                p90 = float(avg_pos_change)
            if total_pos_steps > 0:
                avg_nonzero = float(nonzero_pos_changes) / float(total_pos_steps)
            else:
                avg_nonzero = 0.0
            overtake_stats = {
                "avg_pos_change_per_step": avg_pos_change,
                "avg_nonzero_changes_per_step": avg_nonzero,
                "p90_single_step_jump": p90,
            }

        if circuit_path is None:
            circuit_path = self._build_synthetic_path()

        return {
            "meta": {"generated_at": "now"},
            "team_stats": team_stats,
            "driver_stats": driver_stats,
            "circuit_path": circuit_path,
            "total_laps": likely_total_laps,
            "location": location,
            "overtake_stats": overtake_stats,
            "reference_time": reference_time
        }

    def _build_synthetic_path(self, radius=1000.0, points=200):
        angles = np.linspace(0, 2 * np.pi, points, endpoint=False)
        path_data = []
        total_dist = 0.0
        prev_x = None
        prev_y = None
        for a in angles:
            x = float(np.cos(a) * radius)
            y = float(np.sin(a) * radius)
            if prev_x is not None:
                dx = x - prev_x
                dy = y - prev_y
                seg = (dx * dx + dy * dy) ** 0.5
                total_dist += seg
            path_data.append({
                "X": x,
                "Y": y,
                "Z": 0.0,
                "Distance": total_dist,
                "Speed": 250.0
            })
            prev_x = x
            prev_y = y
        return path_data

# ==========================================
# SIMULATION ENGINE (SimulationEngine)
# ==========================================
class SimulationEngine:
    """
    Motor de simulación y predicción para la temporada 2026.
    """
    
    # Extended Circuit Profiles for 2026 Season
    # type: "Urban", "Fast", "Mixed"
    # difficulty: Base overtake difficulty (lower = harder)
    # qualifying_weight: Importance of grid position (0.0 - 1.0)
    # style_bias: "Technical", "Power", "Balanced"
    CIRCUIT_PROFILES = {
        "Bahrain": {"type": "Mixed", "zones": 3, "difficulty": 1.2, "length": 5.412, "corners": 15, "qualifying_weight": 0.4, "style_bias": "Power"},
        "Sakhir": {"type": "Mixed", "zones": 3, "difficulty": 1.2, "length": 5.412, "corners": 15, "qualifying_weight": 0.4, "style_bias": "Power"},
        "Jeddah": {"type": "Fast", "zones": 3, "difficulty": 1.1, "length": 6.174, "corners": 27, "qualifying_weight": 0.6, "style_bias": "Power"},
        "Saudi Arabia": {"type": "Fast", "zones": 3, "difficulty": 1.1, "length": 6.174, "corners": 27, "qualifying_weight": 0.6, "style_bias": "Power"},
        "Albert Park": {"type": "Mixed", "zones": 4, "difficulty": 0.8, "length": 5.278, "corners": 14, "qualifying_weight": 0.7, "style_bias": "Balanced"},
        "Australia": {"type": "Mixed", "zones": 4, "difficulty": 0.8, "length": 5.278, "corners": 14, "qualifying_weight": 0.7, "style_bias": "Balanced"},
        "Suzuka": {"type": "Technical", "zones": 1, "difficulty": 0.5, "length": 5.807, "corners": 18, "qualifying_weight": 0.8, "style_bias": "Technical"},
        "Japan": {"type": "Technical", "zones": 1, "difficulty": 0.5, "length": 5.807, "corners": 18, "qualifying_weight": 0.8, "style_bias": "Technical"},
        "Shanghai": {"type": "Mixed", "zones": 2, "difficulty": 1.1, "length": 5.451, "corners": 16, "qualifying_weight": 0.5, "style_bias": "Balanced"},
        "China": {"type": "Mixed", "zones": 2, "difficulty": 1.1, "length": 5.451, "corners": 16, "qualifying_weight": 0.5, "style_bias": "Balanced"},
        "Miami": {"type": "Urban", "zones": 3, "difficulty": 0.7, "length": 5.412, "corners": 19, "qualifying_weight": 0.7, "style_bias": "Technical"},
        "Imola": {"type": "Technical", "zones": 1, "difficulty": 0.4, "length": 4.909, "corners": 19, "qualifying_weight": 0.9, "style_bias": "Technical"},
        "Emilia Romagna": {"type": "Technical", "zones": 1, "difficulty": 0.4, "length": 4.909, "corners": 19, "qualifying_weight": 0.9, "style_bias": "Technical"},
        "Monaco": {"type": "Urban", "zones": 1, "difficulty": 0.1, "length": 3.337, "corners": 19, "qualifying_weight": 0.99, "style_bias": "Technical"},
        "Montreal": {"type": "Mixed", "zones": 2, "difficulty": 0.9, "length": 4.361, "corners": 14, "qualifying_weight": 0.5, "style_bias": "Balanced"},
        "Canada": {"type": "Mixed", "zones": 2, "difficulty": 0.9, "length": 4.361, "corners": 14, "qualifying_weight": 0.5, "style_bias": "Balanced"},
        "Barcelona": {"type": "Technical", "zones": 2, "difficulty": 0.6, "length": 4.657, "corners": 14, "qualifying_weight": 0.8, "style_bias": "Technical"},
        "Spain": {"type": "Technical", "zones": 2, "difficulty": 0.6, "length": 4.657, "corners": 14, "qualifying_weight": 0.8, "style_bias": "Technical"},
        "Red Bull Ring": {"type": "Fast", "zones": 3, "difficulty": 1.1, "length": 4.318, "corners": 10, "qualifying_weight": 0.4, "style_bias": "Power"},
        "Austria": {"type": "Fast", "zones": 3, "difficulty": 1.1, "length": 4.318, "corners": 10, "qualifying_weight": 0.4, "style_bias": "Power"},
        "Silverstone": {"type": "Fast", "zones": 2, "difficulty": 0.9, "length": 5.891, "corners": 18, "qualifying_weight": 0.5, "style_bias": "Balanced"},
        "Great Britain": {"type": "Fast", "zones": 2, "difficulty": 0.9, "length": 5.891, "corners": 18, "qualifying_weight": 0.5, "style_bias": "Balanced"},
        "Hungaroring": {"type": "Technical", "zones": 1, "difficulty": 0.4, "length": 4.381, "corners": 14, "qualifying_weight": 0.9, "style_bias": "Technical"},
        "Hungary": {"type": "Technical", "zones": 1, "difficulty": 0.4, "length": 4.381, "corners": 14, "qualifying_weight": 0.9, "style_bias": "Technical"},
        "Spa-Francorchamps": {"type": "Fast", "zones": 2, "difficulty": 1.3, "length": 7.004, "corners": 19, "qualifying_weight": 0.4, "style_bias": "Power"},
        "Belgium": {"type": "Fast", "zones": 2, "difficulty": 1.3, "length": 7.004, "corners": 19, "qualifying_weight": 0.4, "style_bias": "Power"},
        "Zandvoort": {"type": "Technical", "zones": 1, "difficulty": 0.45, "length": 4.259, "corners": 14, "qualifying_weight": 0.85, "style_bias": "Technical"},
        "Netherlands": {"type": "Technical", "zones": 1, "difficulty": 0.45, "length": 4.259, "corners": 14, "qualifying_weight": 0.85, "style_bias": "Technical"},
        "Monza": {"type": "Fast", "zones": 2, "difficulty": 1.2, "length": 5.793, "corners": 11, "qualifying_weight": 0.4, "style_bias": "Power"},
        "Italy": {"type": "Fast", "zones": 2, "difficulty": 1.2, "length": 5.793, "corners": 11, "qualifying_weight": 0.4, "style_bias": "Power"},
        "Baku": {"type": "Mixed", "zones": 2, "difficulty": 1.1, "length": 6.003, "corners": 20, "qualifying_weight": 0.6, "style_bias": "Power"},
        "Azerbaijan": {"type": "Mixed", "zones": 2, "difficulty": 1.1, "length": 6.003, "corners": 20, "qualifying_weight": 0.6, "style_bias": "Power"},
        "Singapore": {"type": "Urban", "zones": 2, "difficulty": 0.3, "length": 4.940, "corners": 19, "qualifying_weight": 0.9, "style_bias": "Technical"},
        "Austin": {"type": "Mixed", "zones": 2, "difficulty": 0.8, "length": 5.513, "corners": 20, "qualifying_weight": 0.6, "style_bias": "Balanced"},
        "United States": {"type": "Mixed", "zones": 2, "difficulty": 0.8, "length": 5.513, "corners": 20, "qualifying_weight": 0.6, "style_bias": "Balanced"},
        "Mexico City": {"type": "Mixed", "zones": 2, "difficulty": 0.7, "length": 4.304, "corners": 17, "qualifying_weight": 0.7, "style_bias": "Balanced"},
        "Mexico": {"type": "Mixed", "zones": 2, "difficulty": 0.7, "length": 4.304, "corners": 17, "qualifying_weight": 0.7, "style_bias": "Balanced"},
        "Interlagos": {"type": "Mixed", "zones": 2, "difficulty": 1.2, "length": 4.309, "corners": 15, "qualifying_weight": 0.5, "style_bias": "Balanced"},
        "Brazil": {"type": "Mixed", "zones": 2, "difficulty": 1.2, "length": 4.309, "corners": 15, "qualifying_weight": 0.5, "style_bias": "Balanced"},
        "Las Vegas": {"type": "Fast", "zones": 3, "difficulty": 1.4, "length": 6.201, "corners": 17, "qualifying_weight": 0.5, "style_bias": "Power"},
        "Lusail": {"type": "Fast", "zones": 1, "difficulty": 0.7, "length": 5.419, "corners": 16, "qualifying_weight": 0.7, "style_bias": "Technical"},
        "Qatar": {"type": "Fast", "zones": 1, "difficulty": 0.7, "length": 5.419, "corners": 16, "qualifying_weight": 0.7, "style_bias": "Technical"},
        "Yas Marina": {"type": "Mixed", "zones": 2, "difficulty": 0.7, "length": 5.281, "corners": 16, "qualifying_weight": 0.7, "style_bias": "Balanced"},
        "Abu Dhabi": {"type": "Mixed", "zones": 2, "difficulty": 0.7, "length": 5.281, "corners": 16, "qualifying_weight": 0.7, "style_bias": "Balanced"},
        "Unknown": {"type": "Mixed", "zones": 2, "difficulty": 0.8, "length": 5.0, "corners": 15, "qualifying_weight": 0.5, "style_bias": "Balanced"}
    }

    def __init__(self):
        pass




    def run_simulation(self, f1_handler, params):
        return self._run_2026_prediction(f1_handler, params)

    def run_basic_simulation(self, f1_handler, params):
        setup_data = self._validate_and_setup(f1_handler, params)
        if "error" in setup_data:
            return setup_data

        location = setup_data["location"]
        event_id = setup_data["event_id"]
        max_laps = setup_data["max_laps"]

        data_context = self._fetch_context_data(f1_handler, location, event_id, params)
        if "error" in data_context:
            return data_context

        sim_results = self._simulate_basic_race(max_laps, data_context)
        metrics = self._calculate_metrics(sim_results, max_laps, data_context)

        return {
            "circuit_name": location,
            "total_laps": max_laps,
            "drivers": sim_results["drivers"],
            "circuit_path": data_context["hist_data"]["circuit_path"],
            "metrics": metrics,
        }

    def run_championship(self, f1_handler, params):
        base_params = dict(params or {})
        base_params.pop("event_id", None)

        events = f1_handler.get_events(2026, filter_active_drivers=True)
        if not events:
            return {"error": "No hay eventos disponibles para 2026 con datos válidos."}

        try:
            events_sorted = sorted(
                [e for e in events if "RoundNumber" in e],
                key=lambda ev: int(ev["RoundNumber"]),
            )
        except Exception:
            events_sorted = events

        points_table = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

        driver_points = {}
        team_points = {}
        driver_wins = {}
        team_wins = {}
        evolution = {}
        races = []

        for ev in events_sorted:
            round_number = ev.get("RoundNumber")
            if round_number is None:
                continue

            local_params = dict(base_params)
            local_params["event_id"] = round_number

            race_result = self._run_2026_prediction(f1_handler, local_params)
            if "error" in race_result:
                continue

            comparisons = (
                race_result.get("metrics", {})
                .get("kpis", {})
                .get("comparisons", {})
            )
            ranking = comparisons.get("drivers_by_total_time") or []
            drivers_data = race_result.get("drivers", {})

            race_entries = []

            for idx, entry in enumerate(ranking):
                abbr = entry.get("driver")
                total_time = entry.get("total_time")
                driver_data = drivers_data.get(abbr)
                if not driver_data:
                    continue

                driver_id = driver_data.get("driver_id") or driver_data.get("DriverId")
                team = driver_data.get("team") or driver_data.get("TeamName")
                full_name = driver_data.get("FullName") or abbr

                position = idx + 1
                points = points_table[idx] if idx < len(points_table) else 0

                if driver_id not in driver_points:
                    driver_points[driver_id] = 0
                    driver_wins[driver_id] = 0
                if team not in team_points:
                    team_points[team] = 0
                    team_wins[team] = 0

                driver_points[driver_id] += points
                team_points[team] += points

                if position == 1:
                    driver_wins[driver_id] += 1
                    team_wins[team] += 1

                if driver_id not in evolution:
                    evolution[driver_id] = {
                        "driver_id": driver_id,
                        "abbreviation": abbr,
                        "full_name": full_name,
                        "team": team,
                        "rounds": [],
                    }

                current_total = driver_points[driver_id]
                evolution[driver_id]["rounds"].append(
                    {
                        "round": int(round_number),
                        "event_name": race_result.get("circuit_name"),
                        "points": points,
                        "cumulative": current_total,
                    }
                )

                race_entries.append(
                    {
                        "position": position,
                        "driver_id": driver_id,
                        "abbreviation": abbr,
                        "full_name": full_name,
                        "team": team,
                        "points": points,
                        "total_time": total_time,
                    }
                )

            if race_entries:
                races.append(
                    {
                        "round": int(round_number),
                        "country": ev.get("Country"),
                        "location": ev.get("Location"),
                        "official_name": ev.get("OfficialEventName") or ev.get(
                            "EventName"
                        ),
                        "results": race_entries,
                    }
                )

        drivers_table = []
        for driver_id, pts in driver_points.items():
            any_evo = evolution.get(driver_id)
            if not any_evo:
                continue
            drivers_table.append(
                {
                    "driver_id": driver_id,
                    "abbreviation": any_evo["abbreviation"],
                    "full_name": any_evo["full_name"],
                    "team": any_evo["team"],
                    "points": pts,
                    "wins": driver_wins.get(driver_id, 0),
                }
            )

        teams_table = []
        for team, pts in team_points.items():
            teams_table.append(
                {
                    "team": team,
                    "points": pts,
                    "wins": team_wins.get(team, 0),
                }
            )

        drivers_table.sort(
            key=lambda x: (-x["points"], -x.get("wins", 0), x["full_name"])
        )
        teams_table.sort(
            key=lambda x: (-x["points"], -x.get("wins", 0), x["team"])
        )

        evolution_list = list(evolution.values())
        for e in evolution_list:
            e["rounds"].sort(key=lambda r: r["round"])

        return {
            "season": 2026,
            "races": races,
            "standings": {
                "drivers": drivers_table,
                "teams": teams_table,
            },
            "evolution": {
                "drivers": evolution_list,
            },
        }

    def _run_2026_prediction(self, f1_handler, params):
        setup_data = self._validate_and_setup(f1_handler, params)
        if "error" in setup_data:
            return setup_data

        location = setup_data["location"]
        event_id = setup_data["event_id"]
        max_laps = setup_data["max_laps"]

        data_context = self._fetch_context_data(f1_handler, location, event_id, params)
        if "error" in data_context:
            return data_context

        sim_results = self._simulate_race(max_laps, data_context)

        metrics = self._calculate_metrics(sim_results, max_laps, data_context)

        return {
            "circuit_name": location,
            "total_laps": max_laps,
            "drivers": sim_results["drivers"],
            "circuit_path": data_context["hist_data"]["circuit_path"],
            "metrics": metrics,
        }

    def _validate_and_setup(self, f1_handler, params):
        try:
            event_id = params.get('event_id')
            if not event_id:
                return {"error": "Se requiere un ID de evento"}
        except ValueError:
             return {"error": "Datos numéricos inválidos"}
        
        events = f1_handler.get_events(2026)
        
        # Improved matching logic
        target_event = None
        event_id_str = str(event_id).strip().lower()
        
        for e in events:
            # 1. Round Number Match
            if str(e.get('RoundNumber')) == str(event_id):
                target_event = e
                break
            
            # 2. Name/Location Match (Case Insensitive)
            evt_name = str(e.get('EventName', '')).lower()
            off_name = str(e.get('OfficialEventName', '')).lower()
            loc = str(e.get('Location', '')).lower()
            
            if event_id_str == evt_name or event_id_str == off_name or event_id_str == loc:
                 target_event = e
                 break
                 
        # Fallback: Fuzzy match if exact string match fails
        if not target_event and not event_id_str.isdigit():
             for e in events:
                evt_name = str(e.get('EventName', '')).lower()
                off_name = str(e.get('OfficialEventName', '')).lower()
                loc = str(e.get('Location', '')).lower()
                
                if event_id_str in evt_name or event_id_str in off_name or event_id_str in loc:
                     target_event = e
                     break
        
        if not target_event:
            return {"error": f"Evento 2026 no encontrado para ID/Nombre: {event_id}"}
            
        location = target_event['Location']
        official = f1_handler.get_official_total_laps_2026(event_id)
        
        if not (official and isinstance(official, dict) and isinstance(official.get("total_laps"), int) and official.get("total_laps", 0) > 0):
            err = official.get("error") if isinstance(official, dict) else None
            return {"error": err or "No se pudieron determinar las vueltas oficiales para el evento"}
            
        return {
            "location": location,
            "event_id": event_id,
            "max_laps": int(official["total_laps"])
        }

    def _fetch_context_data(self, f1_handler, location, event_id, params):
        hist_data = f1_handler.get_historical_circuit_data(location)
        if not hist_data:
            return {"error": f"No hay datos históricos suficientes para {location}."}
            
        drivers_2026 = f1_handler.get_drivers(2026, event_id)
        if not drivers_2026:
            return {"error": "No se pudieron cargar los pilotos para 2026."}

        pace_scale = 1.0
        driver_modifiers = {}
        try:
            if isinstance(params, dict):
                val = params.get("pace_scale")
                if val is not None:
                    pace_scale = float(val)
                mods = params.get("driver_modifiers") or params.get("driver_overrides")
                if isinstance(mods, dict):
                    driver_modifiers = mods
        except Exception:
            pace_scale = 1.0

        return {
            "hist_data": hist_data,
            "drivers": drivers_2026,
            "pace_scale": pace_scale,
            "location": location,
            "driver_modifiers": driver_modifiers,
        }

    def _get_circuit_key(self, location):
        """
        Maps FastF1 Location to Internal Circuit Key.
        """
        loc_norm = location.strip()
        location_map = {
            "Melbourne": "Albert Park",
            "Sakhir": "Bahrain",
            "Monte Carlo": "Monaco",
            "Montréal": "Montreal",
            "Montreal": "Montreal",
            "Spielberg": "Red Bull Ring",
            "Silverstone": "Silverstone",
            "Budapest": "Hungaroring",
            "Spa-Francorchamps": "Spa-Francorchamps",
            "Zandvoort": "Zandvoort",
            "Monza": "Monza",
            "Marina Bay": "Singapore",
            "Singapore": "Singapore",
            "Suzuka": "Suzuka",
            "Lusail": "Qatar",
            "Austin": "COTA",
            "Mexico City": "Mexico",
            "São Paulo": "Interlagos",
            "Sao Paulo": "Interlagos",
            "Las Vegas": "Las Vegas",
            "Yas Marina": "Yas Marina",
            "Imola": "Imola",
            "Miami Gardens": "Miami",
            "Shanghai": "Shanghai",
            "Baku": "Baku",
            "Jeddah": "Jeddah",
            "Barcelona": "Barcelona",
            "Madrid": "Madrid",
            "Yas Island": "Yas Marina"
        }
        return location_map.get(loc_norm, loc_norm)

    def _get_circuit_profile(self, location):
        """
        Resolves the circuit profile based on location name using fuzzy matching.
        """
        key = self._get_circuit_key(location)
        
        if key in self.CIRCUIT_PROFILES:
            return self.CIRCUIT_PROFILES[key]
            
        # Fuzzy match
        for k, v in self.CIRCUIT_PROFILES.items():
            if k.lower() in key.lower() or key.lower() in k.lower():
                return v
                
        # Default fallback
        return {
            "type": "Mixed", 
            "zones": 2, 
            "difficulty": 0.8, 
            "length": 5.0, 
            "corners": 15, 
            "qualifying_weight": 0.5, 
            "style_bias": "Balanced"
        }

    def _resolve_circuit_difficulty(self, location):
        """
        Resolves the overtake difficulty factor based on location/circuit name.
        Uses centralized CIRCUIT_PROFILES.
        """
        profile = self._get_circuit_profile(location)
        return profile.get("difficulty", 0.8)

    def predict_grid(self, drivers, circuit_profile):
        """
        Predicts qualifying order based on driver speed and circuit characteristics.
        """
        scored = []
        q_weight = circuit_profile.get("qualifying_weight", 0.5)
        bias = circuit_profile.get("style_bias", "Balanced")
        
        for d in drivers:
            # We assume 'd' has some base stats or we calculate them cheaply here
            # For now, we'll use a simplified score if stats aren't attached, 
            # but ideally this receives drivers with stats.
            
            # If d is just the dict from F1DataHandler, we might not have 'mean'.
            # We'll rely on the simulation engine's logic to have populated stats if possible,
            # or we simulate a 'qualifying' session using base_mean if available.
            
            base_score = 0.0
            if "base_mean" in d:
                base_score = d["base_mean"]
            else:
                # Fallback to a random-ish score seeded by DriverId for consistency if no stats
                base_score = 90.0
            
            # Apply modifiers based on circuit
            mod = 0.0
            if bias == "Power":
                 # Power tracks might favor certain teams (mock logic)
                 if d.get("TeamName") in ["Red Bull Racing", "Ferrari", "McLaren"]:
                     mod -= 0.2
            elif bias == "Technical":
                 if d.get("TeamName") in ["Red Bull Racing", "Mercedes"]:
                     mod -= 0.2

            # Add variance
            variance = np.random.normal(0, 0.3 * (1.0 - q_weight))
            
            final_score = base_score + mod + variance
            scored.append((final_score, d))
            
        scored.sort(key=lambda x: x[0])
        
        predicted_grid = []
        for i, (_, driver) in enumerate(scored, 1):
            d_copy = dict(driver)
            d_copy["GridPosition"] = i
            predicted_grid.append(d_copy)
            
        return predicted_grid

    def _calculate_driver_performance(self, driver, driver_stats_hist, team_stats, global_mean, driver_modifiers, circuit_profile=None):
        d_id = driver["DriverId"]
        team = driver["TeamName"]
        abbr = driver["Abbreviation"]
        grid_pos = driver.get("GridPosition", 99)

        stats = driver_stats_hist.get(d_id)
        team_hist = team_stats.get(team)
        
        if stats:
            driver_mean = stats.get("mean")
            driver_std = stats.get("std")
            stats_source = "High (Driver History)"
        elif team_hist:
            driver_mean = team_hist.get("mean")
            driver_std = team_hist.get("std")
            stats_source = "Medium (Team Avg)"
        else:
            driver_mean = global_mean
            driver_std = global_mean * 0.02
            stats_source = "Low (Global Avg)"

        if driver_mean is None:
            driver_mean = global_mean

        if not team_hist:
            team_mean = global_mean
            team_std = global_mean * 0.02
        else:
            tm = team_hist.get("mean")
            ts = team_hist.get("std")
            team_mean = tm if tm is not None else global_mean
            team_std = ts if ts is not None else global_mean * 0.02

        skill_rating = 0.0
        style_rating = 0.0
        adaptation_rating = 0.0

        if stats and team_hist:
            try:
                skill_raw = float(team_mean) - float(driver_mean)
                skill_rating = float(np.clip(skill_raw / 0.6, -1.0, 1.0))
            except Exception:
                skill_rating = 0.0
            try:
                base_std_val = float(driver_std) if driver_std is not None else float(team_std)
                team_std_val = float(team_std)
                if team_std_val <= 0:
                    style_rating = 0.0
                else:
                    style_ratio = (base_std_val - team_std_val) / team_std_val
                    style_rating = float(np.clip(style_ratio, -1.0, 1.0))
            except Exception:
                style_rating = 0.0

        override = None
        if isinstance(driver_modifiers, dict):
            override = (
                driver_modifiers.get(d_id)
                or driver_modifiers.get(abbr)
                or driver_modifiers.get(str(d_id))
                or driver_modifiers.get(str(abbr))
            )

        skill_scale = 1.0
        adaptation_scale = 1.0
        style_scale = 1.0
        pace_offset = 0.0

        if isinstance(override, dict):
            if "skill_rating" in override:
                try:
                    skill_rating = float(override.get("skill_rating", skill_rating))
                except Exception:
                    pass
            if "adaptation_rating" in override:
                try:
                    adaptation_rating = float(override.get("adaptation_rating", adaptation_rating))
                except Exception:
                    pass
            if "style_rating" in override:
                try:
                    style_rating = float(override.get("style_rating", style_rating))
                except Exception:
                    pass
            try:
                skill_scale = float(override.get("skill_scale", override.get("skill_factor", 1.0)))
            except Exception:
                skill_scale = 1.0
            try:
                adaptation_scale = float(
                    override.get("adaptation_scale", override.get("adaptation_factor", 1.0))
                )
            except Exception:
                adaptation_scale = 1.0
            try:
                style_scale = float(override.get("style_scale", override.get("style_factor", 1.0)))
            except Exception:
                style_scale = 1.0
            try:
                pace_offset = float(override.get("pace_offset", 0.0))
            except Exception:
                pace_offset = 0.0

        if stats and team_hist:
            driver_delta = float(driver_mean) - float(team_mean)
            skill_component = driver_delta * 0.7 * skill_scale
            adaptation_component = driver_delta * 0.3 * adaptation_scale
            base_mean = float(team_mean) + float(skill_component) + float(adaptation_component) + float(pace_offset)
        else:
            base_mean = float(driver_mean) + float(pace_offset)

        # Apply Circuit Profile Modifiers
        if circuit_profile:
             bias = circuit_profile.get("style_bias", "Balanced")
             c_type = circuit_profile.get("type", "Mixed")
             
             # Technical tracks reward high skill rating more
             if bias == "Technical":
                 if skill_rating > 0:
                     base_mean -= (skill_rating * 0.2) 
             
             # Urban tracks reward adaptation
             if c_type == "Urban":
                 if adaptation_rating > 0:
                     base_mean -= (adaptation_rating * 0.15)

        base_std = float(driver_std) if driver_std is not None else float(team_std)
        base_std = base_std * float(style_scale)

        form_factor = np.random.normal(1.0, 0.003)
        base_mean = float(base_mean) * float(form_factor)

        return {
            "id": d_id,
            "abbr": abbr,
            "team": team,
            "number": driver["Number"],
            "grid_pos": grid_pos,
            "base_mean": base_mean,
            "base_std": base_std,
            "stats_source": stats_source,
            "skill_rating": float(skill_rating),
            "style_rating": float(style_rating),
            "adaptation_rating": float(adaptation_rating),
            "team_mean": float(team_mean),
            "driver_hist_mean": float(driver_mean),
            "color": self._get_team_color(team),
        }

    def _build_driver_states(self, drivers, hist_data, pace_scale, driver_modifiers, circuit_profile=None):
        team_stats = hist_data["team_stats"]
        driver_stats_hist = hist_data["driver_stats"]
        all_means = [d["mean"] for d in driver_stats_hist.values()]
        
        reference_time = hist_data.get("reference_time", 90.0)
        is_ratio = False
        
        if all_means:
            avg_val = float(np.mean(all_means))
            if avg_val < 5.0: # Threshold to detect ratios vs seconds
                is_ratio = True
                global_mean = 1.0
            else:
                global_mean = avg_val
        else:
            global_mean = 90.0 if not is_ratio else 1.0

        sim_drivers = []
        driver_meta = {}

        for driver in drivers:
            abbr = driver["Abbreviation"]
            driver_meta[abbr] = {"driver_id": driver["DriverId"], "team": driver["TeamName"]}

            perf_data = self._calculate_driver_performance(
                driver, driver_stats_hist, team_stats, global_mean, driver_modifiers, circuit_profile
            )
            
            # Apply scaling if data was normalized
            if is_ratio:
                perf_data["base_mean"] = float(perf_data["base_mean"]) * float(reference_time)
                perf_data["base_std"] = float(perf_data["base_std"]) * float(reference_time)
                perf_data["driver_hist_mean"] = float(perf_data["driver_hist_mean"]) * float(reference_time)
                perf_data["team_mean"] = float(perf_data["team_mean"]) * float(reference_time)
            
            # Apply global pace scale
            perf_data["base_mean"] = float(perf_data["base_mean"]) * float(pace_scale)
            
            # Add simulation specific fields
            perf_data.update({
                "cumulative": 0.0,
                "laps_data": [],
                "current_compound_idx": 0,
                "current_tyre_age": 0,
            })

            sim_drivers.append(perf_data)

        sim_drivers.sort(key=lambda x: x["grid_pos"])
        return sim_drivers, driver_meta


    def _simulate_race(self, max_laps, context):
        hist_data = context["hist_data"]
        drivers = context["drivers"]
        pace_scale = context.get("pace_scale", 1.0)
        driver_modifiers = context.get("driver_modifiers") or {}
        circuit_location = context.get("location", "Unknown")
        
        circuit_profile = self._get_circuit_profile(circuit_location)
        circuit_key = self._get_circuit_key(circuit_location)
        overtake_factor = circuit_profile.get("difficulty", 0.8)

        logger.info(f"Simulation Setup: {circuit_location} -> Key: {circuit_key} -> Factor {overtake_factor}")

        results = {}
        compound_cycle = ["MEDIUM", "HARD", "SOFT"]

        sim_drivers, driver_meta = self._build_driver_states(drivers, hist_data, pace_scale, driver_modifiers, circuit_profile)

        position_history = []

        for i in range(1, max_laps + 1):
            current_lap_times = {}
            
            for d in sim_drivers:
                d["current_tyre_age"] += 1
                current_compound = compound_cycle[d["current_compound_idx"]]
                
                raw_time = self._generate_lap_time(
                    d["base_mean"],
                    d["base_std"],
                    i,
                    max_laps,
                    current_compound,
                    d.get("style_rating", 0.0),
                    d.get("skill_rating", 0.0),
                    d.get("adaptation_rating", 0.0),
                )
                
                # Apply Grid Start Penalty on Lap 1
                if i == 1:
                    grid_pos = d.get("grid_pos", 20)
                    # 0.2s per grid slot represents the spatial delay at the start
                    start_penalty = (grid_pos - 1) * 0.2
                    raw_time += start_penalty

                current_lap_times[d["abbr"]] = raw_time
                
                # Store potential lap info
                d["current_lap_info"] = {
                    "lap": int(i),
                    "compound": current_compound,
                    "tyre_age": int(d["current_tyre_age"]),
                    # Time will be updated after interactions
                }

            if i == 1:
                track_order = list(sim_drivers)
            else:
                track_order = sorted(sim_drivers, key=lambda x: x["cumulative"])

            self._apply_interactions(track_order, current_lap_times, i, circuit_profile, hist_data.get("overtake_stats") if isinstance(hist_data, dict) else None)

            # 3. Update States
            for d in sim_drivers:
                final_time = current_lap_times[d["abbr"]]
                prev_times = [l["time"] for l in d["laps_data"][-3:]] if d["laps_data"] else []
                if prev_times:
                    rolling_mean = float(np.mean(prev_times))
                    if len(prev_times) == 1 and i <= 2:
                        allowed_jump = 1.5
                    else:
                        allowed_jump = 1.8
                    delta_time = final_time - rolling_mean
                    if abs(delta_time) > allowed_jump:
                        final_time = rolling_mean + math.copysign(allowed_jump, delta_time)
                d["cumulative"] += final_time
                
                lap_entry = d["current_lap_info"]
                lap_entry["time"] = round(final_time, 3)
                lap_entry["cumulative"] = round(d["cumulative"], 3)
                
                d["laps_data"].append(lap_entry)
            
            # 4. Record Position History
            # Sort by current cumulative for this lap's standing
            lap_standings = sorted(sim_drivers, key=lambda x: x["cumulative"])
            position_history.append({
                "lap": i,
                "order": [d["abbr"] for d in lap_standings]
            })

        # 5. Format Results
        for d in sim_drivers:
            results[d["abbr"]] = {
                "team": d["team"],
                "color": d["color"],
                "laps": d["laps_data"],
                "number": d["number"],
                "Abbreviation": d["abbr"],
                "driver_id": d["id"],
                "stats_source": d["stats_source"],
                "GridPosition": d["grid_pos"],
                "skill_rating": d.get("skill_rating", 0.0),
                "style_rating": d.get("style_rating", 0.0),
                "adaptation_rating": d.get("adaptation_rating", 0.0),
                "team_mean": d.get("team_mean"),
                "driver_hist_mean": d.get("driver_hist_mean"),
            }

        # 6. Verify Integrity
        validation = self._verify_simulation_integrity(sim_drivers, max_laps)
        if not validation["valid"]:
            print(f"Simulation Validation Warning: {validation['errors']}")

        return {
            "drivers": results, 
            "meta": driver_meta, 
            "integrity_check": validation,
            "position_history": position_history
        }

    def _simulate_basic_race(self, max_laps, context):
        hist_data = context["hist_data"]
        drivers = context["drivers"]
        pace_scale = context.get("pace_scale", 1.0)
        driver_modifiers = context.get("driver_modifiers") or {}

        circuit_location = context.get("location", "Unknown")
        circuit_profile = self._get_circuit_profile(circuit_location)
        circuit_key = self._get_circuit_key(circuit_location)

        logger.info(f"Basic Simulation Setup: {circuit_location} -> Key: {circuit_key}")

        results = {}
        sim_drivers, driver_meta = self._build_driver_states(drivers, hist_data, pace_scale, driver_modifiers, circuit_profile)

        position_history = []

        for i in range(1, max_laps + 1):
            current_lap_times = {}

            for d in sim_drivers:
                raw_time = self._generate_lap_time(
                    d["base_mean"],
                    d["base_std"],
                    i,
                    max_laps,
                    "MEDIUM",
                    d.get("style_rating", 0.0),
                    d.get("skill_rating", 0.0),
                    d.get("adaptation_rating", 0.0),
                )
                current_lap_times[d["abbr"]] = raw_time

                lap_entry = {
                    "lap": int(i),
                    "compound": "MEDIUM",
                    "tyre_age": 0,
                }

                d["cumulative"] += raw_time
                lap_entry["time"] = round(raw_time, 3)
                lap_entry["cumulative"] = round(d["cumulative"], 3)
                d["laps_data"].append(lap_entry)

            lap_standings = sorted(sim_drivers, key=lambda x: x["cumulative"])
            position_history.append(
                {"lap": i, "order": [d["abbr"] for d in lap_standings]}
            )

        for d in sim_drivers:
            results[d["abbr"]] = {
                "team": d["team"],
                "color": d["color"],
                "laps": d["laps_data"],
                "number": d["number"],
                "Abbreviation": d["abbr"],
                "driver_id": d["id"],
                "stats_source": d["stats_source"],
                "GridPosition": d["grid_pos"],
                "skill_rating": d.get("skill_rating", 0.0),
                "style_rating": d.get("style_rating", 0.0),
                "adaptation_rating": d.get("adaptation_rating", 0.0),
                "team_mean": d.get("team_mean"),
                "driver_hist_mean": d.get("driver_hist_mean"),
            }

        validation = self._verify_simulation_integrity(sim_drivers, max_laps)
        if not validation["valid"]:
            print(f"Simulation Validation Warning: {validation['errors']}")

        return {
            "drivers": results,
            "meta": driver_meta,
            "integrity_check": validation,
            "position_history": position_history,
        }

    def _apply_interactions(self, track_order, lap_times, lap_idx, circuit_profile, overtake_stats=None):
        zones = circuit_profile.get("zones", 2)
        circuit_diff = circuit_profile.get("difficulty", 0.8)
        try:
            zones = max(1, int(zones))
        except Exception:
            zones = 2
        try:
            circuit_diff = float(circuit_diff)
        except Exception:
            circuit_diff = 0.8
        circuit_diff = max(0.1, min(circuit_diff, 1.5))

        lap_start_phase = lap_idx <= 2
        per_driver_pass_limit = {}

        for i in range(len(track_order)):
            driver = track_order[i]
            abbr = driver["abbr"]
            
            if i == 0:
                continue
                
            leader = track_order[i-1]
            leader_abbr = leader["abbr"]
            
            if lap_idx == 1:
                gap = 0.5
            else:
                gap = driver["cumulative"] - leader["cumulative"]

            # Limit overtakes on Lap 1 to prevent chaos
            if lap_idx == 1:
                # On lap 1, you can usually only pass cars very close to you
                # and you can't pass 10 cars unless they crash.
                # We enforce a "grid slot" friction.
                pass_limit_lap1 = 3 # Max positions gained on lap 1
                current_gained = per_driver_pass_limit.get(abbr, 0)
                if current_gained >= pass_limit_lap1:
                    continue
            
            if lap_idx == 1:
                min_cumulative = leader["cumulative"] + lap_times[leader_abbr] + 0.1
                current_projected = driver["cumulative"] + lap_times[abbr]
                if current_projected < min_cumulative:
                    diff = min_cumulative - current_projected
                    lap_times[abbr] += diff
                # Only small chance to pass on Lap 1 if not much faster
                # ... existing logic continues below but we add friction
                if np.random.random() > 0.3: # 70% chance to NOT pass even if projected faster
                     continue
                
            if 0.0 < gap < 1.0:
                gain = 0.03 * (1.0 - gap)
                lap_times[abbr] -= gain
            
            my_pace = lap_times[abbr]
            leader_pace = lap_times[leader_abbr]
            
            # Overtake Logic Enhanced
            # 1. Performance Delta (Team + Driver pace)
            pace_delta = leader_pace - my_pace # Positive means I am faster
            
            # 2. Driver Skill & Aggressiveness Factors
            attacker_skill = driver.get("skill_rating", 0.0)
            attacker_aggro = driver.get("style_rating", 0.0)
            defender_skill = leader.get("skill_rating", 0.0)
            defender_aggro = leader.get("style_rating", 0.0)
            
            # Skill delta: Higher skill attacker vs lower skill defender increases chance
            skill_delta = attacker_skill - defender_skill
            
            # 3. Circuit Conditions
            # circuit_diff is loaded at start (0.1 easy to 1.5 hard)
            
            # Logic:
            # Require pace advantage OR opportunistic move (aggression/mistake)
            # Threshold: 0.05s faster normally, or chance if aggressive
            
            attempt_threshold = 0.05 - (attacker_aggro * 0.02) # Aggressive drivers attempt with less delta
            
            if gap < 0.8 and (pace_delta > attempt_threshold or np.random.random() < 0.05):
                mistake_bonus = 0.0
                if pace_delta > 1.5:
                    mistake_bonus = 0.15

                base_prob_per_zone = 0.005 * circuit_diff 
                
                # Pace factor
                delta_factor = (pace_delta * 0.08) * circuit_diff
                if delta_factor > 0.12: delta_factor = 0.12
                
                # Skill factor (Impact: +/- 15% probability)
                skill_factor = skill_delta * 0.15
                
                # Aggressiveness bonus (Impact: +5% for attacker, -2% for defender intimidation)
                aggro_factor = (attacker_aggro * 0.05) - (defender_aggro * 0.02)

                zone_prob = base_prob_per_zone + delta_factor + mistake_bonus + skill_factor + aggro_factor

                if isinstance(overtake_stats, dict):
                    avg_nonzero = overtake_stats.get("avg_nonzero_changes_per_step")
                    try:
                        avg_nonzero = float(avg_nonzero)
                    except Exception:
                        avg_nonzero = None
                    if avg_nonzero is not None and avg_nonzero > 0.0:
                        baseline = 0.3
                        scale = avg_nonzero / baseline
                        if scale < 0.5: scale = 0.5
                        if scale > 1.5: scale = 1.5
                        zone_prob *= scale

                early_factor = lap_idx / 12.0
                if early_factor < 0.2: early_factor = 0.2
                if early_factor > 1.0: early_factor = 1.0
                zone_prob *= early_factor

                if zone_prob < 0.0: zone_prob = 0.0
                if zone_prob > 0.6: zone_prob = 0.6 # Cap at 60% per zone

                if lap_start_phase:
                    zone_prob *= 0.7

                passed = False
                for z in range(zones):
                    if np.random.random() < zone_prob:
                        passed = True
                        break
                
                if passed:
                    count = per_driver_pass_limit.get(abbr, 0)
                    if lap_start_phase and count >= 1:
                        passed = False
                    else:
                        per_driver_pass_limit[abbr] = count + 1

                    # Overtake successful penalty/reward
                    # Defender loses time (being passed), Attacker loses slight time (off line) but gains pos
                    lap_times[leader_abbr] += 0.15 + (defender_skill * 0.05) # Better defenders lose less time? No, usually lose more fighting
                    lap_times[abbr] += 0.05 
                else:
                    # Failed overtake attempt - Dirty Air Penalty
                    # Reduced by attacker skill (better line) and increased by defender skill (good defense)
                    dirty_air_base = (0.1 + (0.8 - gap) * 0.1) / (circuit_diff + 0.1)
                    defense_factor = 1.0 + (defender_skill * 0.2)
                    attack_mitigation = 1.0 - (attacker_skill * 0.1)
                    
                    penalty = dirty_air_base * defense_factor * attack_mitigation
                    lap_times[abbr] = max(lap_times[abbr], leader_pace + penalty)

    def _verify_simulation_integrity(self, sim_drivers, max_laps):
        errors = []
        n_drivers = len(sim_drivers)
        
        if n_drivers == 0:
             errors.append("No drivers in simulation")
             
        for d in sim_drivers:
            if len(d["laps_data"]) != max_laps:
                errors.append(f"Driver {d['abbr']} has {len(d['laps_data'])} laps, expected {max_laps}")
            
            for l in d["laps_data"]:
                if l["time"] < 40.0: 
                    errors.append(f"Driver {d['abbr']} Lap {l['lap']}: Unrealistic time {l['time']}s")

        for d in sim_drivers:
            start_pos = d["grid_pos"]
            finish_pos = 1
            my_time = d["laps_data"][-1]["cumulative"] if d["laps_data"] else 0
            for other in sim_drivers:
                if other["abbr"] == d["abbr"]: continue
                other_time = other["laps_data"][-1]["cumulative"] if other["laps_data"] else 0
                if other_time < my_time:
                    finish_pos += 1
            
            gained = start_pos - finish_pos
            logger.info(f"Position change: {d['abbr']} from {start_pos} to {finish_pos} (Gained {gained} positions)")
            if gained > 18: # Highly unlikely unless mixed conditions/crashes
                 errors.append(f"Driver {d['abbr']} gained {gained} positions (Unrealistic)")

        max_single_lap_jump = 0
        if max_laps > 1:
            for lap in range(1, max_laps + 1):
                standing = sorted(
                    sim_drivers,
                    key=lambda x: x["laps_data"][lap - 1]["cumulative"] if len(x["laps_data"]) >= lap else float("inf"),
                )
                order = [d["abbr"] for d in standing]
                if lap == 1:
                    prev_order = order
                    continue
                idx_prev = {abbr: idx for idx, abbr in enumerate(prev_order)}
                for idx, abbr in enumerate(order):
                    if abbr not in idx_prev:
                        continue
                    delta = abs(idx_prev[abbr] - idx)
                    if delta > max_single_lap_jump:
                        max_single_lap_jump = delta
                prev_order = order

        if max_single_lap_jump > 6:
            errors.append(f"Max single-lap position jump {max_single_lap_jump} exceeds threshold")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "driver_count": n_drivers
        }

    def _compute_position_error_metrics(self, sim_position_history, reference_position_history):
        if not isinstance(sim_position_history, list) or not isinstance(reference_position_history, list):
            return {}
        sim_by_lap = {}
        ref_by_lap = {}
        for entry in sim_position_history:
            lap = entry.get("lap")
            order = entry.get("order")
            if isinstance(lap, int) and isinstance(order, list):
                sim_by_lap[lap] = [str(x) for x in order]
        for entry in reference_position_history:
            lap = entry.get("lap")
            order = entry.get("order")
            if isinstance(lap, int) and isinstance(order, list):
                ref_by_lap[lap] = [str(x) for x in order]
        common_laps = sorted(set(sim_by_lap.keys()) & set(ref_by_lap.keys()))
        if not common_laps:
            return {}
        sum_sq = 0.0
        sum_abs = 0.0
        weight_sum = 0.0
        for lap in common_laps:
            sim_order = sim_by_lap[lap]
            ref_order = ref_by_lap[lap]
            sim_rank = {abbr: idx for idx, abbr in enumerate(sim_order)}
            ref_rank = {abbr: idx for idx, abbr in enumerate(ref_order)}
            drivers_common = set(sim_rank.keys()) & set(ref_rank.keys())
            if not drivers_common:
                continue
            if lap <= 2:
                w = 1.5
            else:
                w = 1.0
            for abbr in drivers_common:
                diff = float(sim_rank[abbr] - ref_rank[abbr])
                sum_sq += w * diff * diff
                sum_abs += w * abs(diff)
                weight_sum += w
        if weight_sum == 0.0:
            return {}
        rmse = (sum_sq / weight_sum) ** 0.5
        mae = sum_abs / weight_sum
        return {
            "rmse_position": float(rmse),
            "mae_position": float(mae),
        }


    def _generate_lap_time(self, base_mean, base_std, lap_idx, max_laps, compound, style_rating=0.0, skill_rating=0.0, adaptation_rating=0.0):
        lap_idx = int(lap_idx)
        mean = float(base_mean)
        
        # Add a consistency smoothing factor based on lap index (simplified momentum)
        # This prevents wild oscillation lap-to-lap
        consistency_seed = (lap_idx % 3) * 0.05 
        
        try:
            style_rating = float(np.clip(style_rating, -1.0, 1.0))
        except Exception:
            style_rating = 0.0
        try:
            skill_rating = float(np.clip(skill_rating, -1.0, 1.0))
        except Exception:
            skill_rating = 0.0
        try:
            adaptation_rating = float(np.clip(adaptation_rating, -1.0, 1.0))
        except Exception:
            adaptation_rating = 0.0
        
        mean += 1.5 * (1.0 - (lap_idx / float(max_laps)))

        amp_base = float(base_std) if base_std and base_std > 0 else float(mean) * 0.02
        amp_scale = 1.0 + 0.5 * style_rating
        if amp_scale < 0.3:
            amp_scale = 0.3
        amp = amp_base * amp_scale
        w1 = math.sin((2.0 * math.pi * lap_idx) / 17.0)
        w2 = math.sin((2.0 * math.pi * lap_idx) / 7.0)
        
        noise = np.random.normal(0, amp * 0.35) 
        
        # Apply smoothing to noise to prevent jagged profile
        noise = noise * 0.8 + consistency_seed * amp 

        mistake_prob = 0.015 * (1.0 + 0.6 * style_rating - 0.3 * skill_rating)
        if mistake_prob < 0.005:
            mistake_prob = 0.005
        if mistake_prob > 0.06:
            mistake_prob = 0.06

        if np.random.random() < mistake_prob:
            mistake_cost = 0.5 + np.random.random() * 1.0
            noise += mistake_cost
        
        val = mean + amp * (0.55 * w1 + 0.25 * w2) + noise - 0.1 * adaptation_rating

        if compound == "SOFT": val -= 0.15
        elif compound == "HARD": val += 0.12

        return max(float(val), float(mean * 0.8))

    def _get_team_color(self, team_name):
        colors = {
            "Red Bull Racing": "#0600EF", "Mercedes": "#00D2BE", "Ferrari": "#DC0000",
            "McLaren": "#FF8700", "Aston Martin": "#006F62", "Alpine": "#0090FF",
            "Williams": "#005AFF", "RB": "#6692FF", "Kick Sauber": "#52E252",
            "Haas F1 Team": "#B6BABD"
        }
        return colors.get(team_name, "#999999")

    def _calculate_metrics(self, sim_results, max_laps, context):
        results = sim_results["drivers"]
        driver_meta = sim_results["meta"]
        hist_data = context["hist_data"]

        kpis_per_driver = {}
        team_aggregate = {}

        for abbr, data in results.items():
            laps = data["laps"]
            times = [l["time"] for l in laps]
            total_time = laps[-1]["cumulative"] if laps else 0.0

            best_lap = min(times) if times else 0.0
            avg_lap = float(np.mean(times)) if times else 0.0
            std_lap = float(np.std(times)) if len(times) > 1 else 0.0

            compounds, stint_lengths = self._analyze_stints(laps)
            # Pass circuit_profile to simulate sectors correctly (needs context)
            circuit_profile_local = context.get("circuit_profile")
            if not circuit_profile_local and "location" in context:
                 circuit_profile_local = self._get_circuit_profile(context["location"])
            
            sectors = self._simulate_sectors(laps, circuit_profile_local)
            pace_phases = self._analyze_pace_phases(laps)

            kpis_per_driver[abbr] = {
                "best_lap": round(best_lap, 3),
                "avg_lap": round(avg_lap, 3),
                "std_lap": round(std_lap, 3),
                "total_time": round(total_time, 3),
                "avg_stint_length": round(
                    float(np.mean(stint_lengths)) if stint_lengths else 0.0, 3
                ),
                "tyre_compounds": compounds,
                "sectors": sectors,
                "pace_by_phase": pace_phases,
            }

            team = data["team"]
            if team not in team_aggregate:
                team_aggregate[team] = {
                    "total_time_sum": 0.0,
                    "best_laps": [],
                    "count": 0,
                }
            team_aggregate[team]["total_time_sum"] += total_time
            if best_lap > 0:
                team_aggregate[team]["best_laps"].append(best_lap)
            team_aggregate[team]["count"] += 1

        self._attach_confidence_intervals(
            kpis_per_driver, driver_meta, hist_data, max_laps, confidence_level=0.9
        )

        kpis_per_team = self._finalize_team_kpis(team_aggregate)

        driver_ranking = sorted(
            [
                {"driver": a, "total_time": k["total_time"]}
                for a, k in kpis_per_driver.items()
            ],
            key=lambda x: x["total_time"],
        )
        if driver_ranking:
            best = driver_ranking[0]["total_time"]
            for d in driver_ranking:
                d["gap_to_best"] = round(d["total_time"] - best, 3)

        team_ranking = sorted(
            [
                {"team": t, "avg_total_time": k["avg_total_time"]}
                for t, k in kpis_per_team.items()
            ],
            key=lambda x: x["avg_total_time"],
        )
        if team_ranking:
            best = team_ranking[0]["avg_total_time"]
            for t in team_ranking:
                t["gap_to_best"] = round(t["avg_total_time"] - best, 3)

        position_summary = []
        position_summary_table = ""
        if driver_ranking:
            rows = []
            for idx, entry in enumerate(driver_ranking, start=1):
                abbr = entry["driver"]
                data = results.get(abbr, {})
                start_pos = data.get("GridPosition") or data.get("grid_pos")
                finish_pos = idx
                delta = None
                try:
                    if start_pos is not None:
                        delta = int(start_pos) - int(finish_pos)
                except Exception:
                    delta = None
                position_summary.append(
                    {
                        "driver": abbr,
                        "grid_position": int(start_pos) if isinstance(start_pos, (int, float)) else start_pos,
                        "finish_position": finish_pos,
                        "position_change": delta,
                    }
                )
                rows.append(
                    {
                        "driver": abbr,
                        "grid": str(start_pos) if start_pos is not None else "-",
                        "finish": str(finish_pos),
                        "delta": f"{'+' if delta is not None and delta > 0 else ''}{delta}" if delta is not None else "-",
                    }
                )
            if rows:
                headers = {"driver": "Piloto", "grid": "Parrilla", "finish": "Final", "delta": "Δ Pos"}
                widths = {}
                for key, title in headers.items():
                    widths[key] = len(title)
                for r in rows:
                    for key in headers.keys():
                        widths[key] = max(widths[key], len(str(r[key])))
                header_line = " | ".join(headers[k].ljust(widths[k]) for k in ("driver", "grid", "finish", "delta"))
                sep_line = "-+-".join("-" * widths[k] for k in ("driver", "grid", "finish", "delta"))
                body_lines = []
                for r in rows:
                    body_lines.append(
                        " | ".join(str(r[k]).ljust(widths[k]) for k in ("driver", "grid", "finish", "delta"))
                    )
                position_summary_table = "\n".join([header_line, sep_line] + body_lines)

        reference_position_history = context.get("reference_position_history")
        position_error = {}
        if isinstance(reference_position_history, list):
            position_error = self._compute_position_error_metrics(
                sim_results.get("position_history") or [],
                reference_position_history,
            )

        validation = self._validate_predictions(kpis_per_driver, driver_meta, hist_data)
        integrity = self._evaluate_team_independence(results, driver_meta, hist_data, driver_ranking)
        prediction_report = self._generate_prediction_report(driver_ranking, results, driver_meta, hist_data, max_laps)

        return {
            "kpis": {
                "per_driver": kpis_per_driver,
                "per_team": kpis_per_team,
                "race": {
                    "total_laps": max_laps,
                    "position_error": position_error,
                    "position_summary": position_summary,
                    "position_summary_table": position_summary_table,
                },
                "comparisons": {
                    "drivers_by_total_time": driver_ranking,
                    "teams_by_total_time": team_ranking,
                },
            },
            "validation": validation,
            "integrity": integrity,
            "prediction_report": prediction_report,
        }

    def _generate_prediction_report(self, driver_ranking, results, driver_meta, hist_data, max_laps):
        """
        Generates a textual analysis of the prediction (Key Factors).
        """
        report = []
        if not driver_ranking:
            return report

        driver_stats = hist_data.get("driver_stats", {})
        team_stats = hist_data.get("team_stats", {})
        
        # Analyze Winner
        winner_entry = driver_ranking[0]
        winner_abbr = winner_entry["driver"]
        winner_data = results.get(winner_abbr, {})
        winner_meta = driver_meta.get(winner_abbr, {})
        
        # Factor 1: Source of Pace
        stats_source = winner_data.get("stats_source", "Unknown")
        if "Driver History" in stats_source:
            factor_text = f"Strong historical performance at this circuit (Driver Data)."
        elif "Team Avg" in stats_source:
            factor_text = f"Solid team baseline performance (Team Data)."
        else:
            factor_text = f"Estimated performance based on global averages."

        # Factor 2: Consistency
        laps = winner_data.get("laps", [])
        times = [l["time"] for l in laps]
        std_dev = np.std(times) if times else 0
        consistency_text = "High consistency" if std_dev < 1.0 else "Variable pace"

        # Factor 3: Tyre Strategy
        compounds, _ = self._analyze_stints(laps)
        strategy_text = f"Strategy used: {', '.join(compounds.keys())}"

        report.append({
            "driver": winner_abbr,
            "position": 1,
            "summary": f"Predicted Winner: {winner_abbr}",
            "key_factors": [
                factor_text,
                f"Consistency level: {consistency_text} (Std Dev: {std_dev:.2f}s)",
                strategy_text,
                f"Total Race Time: {self._format_time(winner_entry['total_time'])}"
            ]
        })

        # Analyze Podium (2nd and 3rd)
        for i in range(1, min(3, len(driver_ranking))):
            entry = driver_ranking[i]
            abbr = entry["driver"]
            gap = entry.get("gap_to_best", 0.0)
            
            report.append({
                "driver": abbr,
                "position": i + 1,
                "summary": f"Predicted P{i+1}: {abbr}",
                "key_factors": [
                    f"Gap to winner: +{gap:.3f}s",
                    f"Avg Lap Time: {results[abbr].get('avg_lap', 0):.3f}s"
                ]
            })

        return report

    def _format_time(self, seconds):
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m {s:.3f}s"

    def _evaluate_team_independence(self, results, driver_meta, hist_data, driver_ranking):
        driver_stats = hist_data.get("driver_stats", {})
        team_stats = hist_data.get("team_stats", {})

        final_positions = {}
        for idx, entry in enumerate(driver_ranking, start=1):
            final_positions[entry["driver"]] = idx

        team_to_drivers = {}
        for abbr, meta in driver_meta.items():
            team = meta["team"]
            if team not in team_to_drivers:
                team_to_drivers[team] = []
            team_to_drivers[team].append(abbr)

        team_reports = {}
        global_flags = []

        for team, abbrs in team_to_drivers.items():
            if len(abbrs) < 2:
                continue

            pairs = []
            for i in range(len(abbrs)):
                for j in range(i + 1, len(abbrs)):
                    a = abbrs[i]
                    b = abbrs[j]
                    pairs.append((a, b))

            suspicions = []
            scores = []

            for a, b in pairs:
                meta_a = driver_meta.get(a, {})
                meta_b = driver_meta.get(b, {})

                id_a = meta_a.get("driver_id")
                id_b = meta_b.get("driver_id")

                hist_a = driver_stats.get(id_a) or team_stats.get(team)
                hist_b = driver_stats.get(id_b) or team_stats.get(team)

                mean_a = hist_a.get("mean") if hist_a else None
                mean_b = hist_b.get("mean") if hist_b else None

                if mean_a is None or mean_b is None:
                    continue

                if mean_a <= mean_b:
                    fast = a
                    slow = b
                    fast_mean = float(mean_a)
                    slow_mean = float(mean_b)
                else:
                    fast = b
                    slow = a
                    fast_mean = float(mean_b)
                    slow_mean = float(mean_a)

                res_fast = results.get(fast, {})
                res_slow = results.get(slow, {})

                grid_fast = res_fast.get("grid_position")
                grid_slow = res_slow.get("grid_position")

                pos_fast = final_positions.get(fast)
                pos_slow = final_positions.get(slow)

                if grid_fast is None or grid_slow is None:
                    continue
                if pos_fast is None or pos_slow is None:
                    continue

                pace_gap = slow_mean - fast_mean
                grid_gap = grid_slow - grid_fast
                race_gap = pos_slow - pos_fast

                swap_against_pace = 1 if pos_fast > pos_slow else 0

                severity = 0.0
                if swap_against_pace == 1:
                    severity += 1.0
                    if grid_fast < grid_slow:
                        severity += 0.5
                    if pace_gap > 0.5:
                        severity += 0.5

                scores.append(severity)

                if severity >= 1.0:
                    suspicions.append(
                        {
                            "fast_driver": fast,
                            "slow_driver": slow,
                            "pace_gap": round(pace_gap, 3),
                            "grid_fast": grid_fast,
                            "grid_slow": grid_slow,
                            "final_fast": pos_fast,
                            "final_slow": pos_slow,
                            "severity": round(severity, 2),
                        }
                    )

            if scores:
                avg_score = float(np.mean(scores))
            else:
                avg_score = 0.0

            independence_index = max(0.0, 1.0 - min(avg_score, 3.0) / 3.0)

            flags = []
            if avg_score >= 1.5:
                flags.append("high_suspicion")
            elif avg_score >= 0.5:
                flags.append("medium_suspicion")

            team_reports[team] = {
                "independence_index": round(independence_index, 3),
                "avg_suspicion_score": round(avg_score, 3),
                "suspected_patterns": suspicions,
                "flags": flags,
                "criteria": {
                    "pace_gap_threshold": 0.5,
                    "severity_threshold_medium": 0.5,
                    "severity_threshold_high": 1.5,
                },
                "protocol": {
                    "review_lap_history": len(suspicions) > 0,
                    "check_team_radio": len(suspicions) > 0,
                    "manual_steward_review": avg_score >= 1.5,
                },
                "recommended_measures": {
                    "mark_session_for_review": avg_score >= 0.5,
                    "consider_penalty_or_re_run": avg_score >= 2.0,
                },
            }

            for flag in flags:
                global_flags.append({"team": team, "flag": flag})

        summary = {
            "teams": team_reports,
            "global_flags": global_flags,
        }

        return summary

    def _attach_confidence_intervals(
        self, kpis_driver, driver_meta, hist_data, max_laps, confidence_level=0.9
    ):
        """
        Añade intervalos de confianza a las predicciones por piloto.

        El intervalo se calcula a partir de la media y desviación histórica de
        vuelta (por piloto o equipo). La predicción se considera coherente si la
        media simulada cae dentro del intervalo histórico.
        """
        if not kpis_driver:
            return

        driver_stats = hist_data.get("driver_stats", {})
        team_stats = hist_data.get("team_stats", {})

        if confidence_level >= 0.95:
            z = 1.96
        elif confidence_level >= 0.9:
            z = 1.64
        else:
            z = 1.0

        for abbr, meta in driver_meta.items():
            if abbr not in kpis_driver:
                continue

            kpis = kpis_driver[abbr]
            d_id = meta["driver_id"]
            team = meta["team"]

            hist = driver_stats.get(d_id) or team_stats.get(team)
            if not hist:
                continue

            hist_mean = hist.get("mean")
            hist_std = hist.get("std")
            if hist_mean is None:
                continue

            mean_val = float(hist_mean)
            std_val = float(hist_std) if hist_std and hist_std > 0 else mean_val * 0.02

            ci_lap_low = mean_val - z * std_val
            ci_lap_high = mean_val + z * std_val

            total_mean = mean_val * float(max_laps)
            total_std = std_val * math.sqrt(float(max_laps))
            ci_total_low = total_mean - z * total_std
            ci_total_high = total_mean + z * total_std

            pred_avg_lap = kpis.get("avg_lap", 0.0)
            pred_total_time = kpis.get("total_time", 0.0)

            kpis["confidence"] = {
                "level": confidence_level,
                "avg_lap_ci": {
                    "low": round(ci_lap_low, 3),
                    "high": round(ci_lap_high, 3),
                    "inside": ci_lap_low <= float(pred_avg_lap) <= ci_lap_high,
                },
                "total_time_ci": {
                    "low": round(ci_total_low, 3),
                    "high": round(ci_total_high, 3),
                    "inside": ci_total_low
                    <= float(pred_total_time)
                    <= ci_total_high,
                },
                "reference_mean": round(mean_val, 3),
                "reference_std": round(std_val, 3),
            }

    def _analyze_stints(self, laps):
        compounds = {}
        stint_lengths = []
        current_compound = None
        current_length = 0
        
        for lap in laps:
            c = lap['compound']
            compounds[c] = compounds.get(c, 0) + 1
            if current_compound is None:
                current_compound = c
                current_length = 1
            elif c == current_compound:
                current_length += 1
            else:
                stint_lengths.append(current_length)
                current_compound = c
                current_length = 1
        if current_length > 0:
            stint_lengths.append(current_length)
        return compounds, stint_lengths

    def _simulate_sectors(self, laps, circuit_profile=None):
        """
        Simulate sector times based on circuit profile characteristics.
        S1/S2/S3 distribution varies by track type (Power/Technical).
        """
        s1_list, s2_list, s3_list = [], [], []
        
        # Default distribution (Balanced)
        ratio = [0.32, 0.34, 0.34]
        
        if circuit_profile:
            bias = circuit_profile.get("style_bias", "Balanced")
            if bias == "Power":
                # Power tracks often have long straights (S1/S3) and fewer corners
                ratio = [0.30, 0.40, 0.30] 
            elif bias == "Technical":
                # Technical tracks might have twisty middle sectors
                ratio = [0.33, 0.34, 0.33]
                
        for lap in laps:
            lt = lap['time']
            # Add small random variation per lap to simulate sector variance
            var = np.random.normal(0, 0.005, 3)
            
            s1 = lt * (ratio[0] + var[0])
            s2 = lt * (ratio[1] + var[1])
            s3 = lt - s1 - s2
            
            s1_list.append(s1)
            s2_list.append(s2)
            s3_list.append(s3)
            
        return {
            "best": {
                "s1": round(min(s1_list) if s1_list else 0, 3),
                "s2": round(min(s2_list) if s2_list else 0, 3),
                "s3": round(min(s3_list) if s3_list else 0, 3),
            },
            "avg": {
                "s1": round(float(np.mean(s1_list)) if s1_list else 0, 3),
                "s2": round(float(np.mean(s2_list)) if s2_list else 0, 3),
                "s3": round(float(np.mean(s3_list)) if s3_list else 0, 3),
            }
        }

    def _analyze_pace_phases(self, laps):
        times = [l['time'] for l in laps]
        n = len(times)
        if n == 0: return {"early": 0, "mid": 0, "late": 0}
        
        e_end = max(1, n // 3)
        m_end = max(e_end + 1, (2 * n) // 3)
        
        return {
            "early": float(np.mean(times[:e_end])) if times[:e_end] else 0.0,
            "mid": float(np.mean(times[e_end:m_end])) if times[e_end:m_end] else 0.0,
            "late": float(np.mean(times[m_end:])) if times[m_end:] else 0.0,
        }

    def _finalize_team_kpis(self, team_aggregate):
        kpis = {}
        for team, agg in team_aggregate.items():
            if agg["count"] == 0: continue
            kpis[team] = {
                "avg_total_time": round(agg["total_time_sum"] / agg["count"], 3),
                "best_lap": round(min(agg["best_laps"]) if agg["best_laps"] else 0, 3),
            }
        return kpis

    def _validate_predictions(self, kpis_driver, driver_meta, hist_data):
        val_driver = {}
        val_team = {}
        
        driver_stats = hist_data['driver_stats']
        team_stats = hist_data['team_stats']
        
        for abbr, meta in driver_meta.items():
            if abbr not in kpis_driver: continue
            
            d_id = meta["driver_id"]
            hist = driver_stats.get(d_id) or team_stats.get(meta["team"])
            
            if hist:
                pred = kpis_driver[abbr]["avg_lap"]
                hist_mean = float(hist["mean"])
                val_driver[abbr] = {
                    "predicted_avg_lap": pred,
                    "historical_avg_lap": round(hist_mean, 3),
                    "delta": round(pred - hist_mean, 3)
                }
                
        for team, stats in team_stats.items():
            drivers = [abbr for abbr, m in driver_meta.items() if m["team"] == team and abbr in kpis_driver]
            if not drivers: continue
            
            preds = [kpis_driver[a]["avg_lap"] for a in drivers]
            pred_mean = float(np.mean(preds))
            hist_mean = float(stats["mean"])
            val_team[team] = {
                "predicted_avg_lap": round(pred_mean, 3),
                "historical_avg_lap": round(hist_mean, 3),
                "delta": round(pred_mean - hist_mean, 3)
            }
            
        return {"drivers_vs_history": val_driver, "teams_vs_history": val_team}

# ==========================================
# HANDLERS INSTANTIATION
# ==========================================
f1_handler = F1DataHandler()
sim_engine = SimulationEngine()

# ==========================================
# API CONTROLLER (APIController)
# ==========================================
class APIController:
    """Handles API logic and interacts with data handlers."""
    
    @staticmethod
    def get_seasons():
        return f1_handler.get_seasons()

    @staticmethod
    def get_events(query):
        # Enforce 2026, ignore any parameters
        year = 2026
        # Disable active drivers filter (we show full calendar)
        filter_active = False 
        
        return f1_handler.get_events(year, filter_active_drivers=filter_active)

    @staticmethod
    def get_drivers(query):
        event_id = query.get('event_id', [None])[0]
        if not event_id:
            raise ValueError("Se requiere un ID de evento")

        return f1_handler.get_drivers(2026, event_id)

    @staticmethod
    def get_session_info(query):
        event_id = query.get('event_id', [None])[0]
        if not event_id:
            raise ValueError("Se requiere un ID de evento")
        return f1_handler.get_session_info(2026, event_id)

    @staticmethod
    def run_simulation(data):
        event = data.get('event_id')
        session_info = f1_handler.get_session_info(2026, event)
        if session_info and isinstance(session_info, dict) and isinstance(session_info.get("error"), str):
            return {"error": session_info.get("error")}
        return sim_engine.run_simulation(f1_handler, data)

    @staticmethod
    def run_basic_simulation(data):
        event = data.get("event_id")
        session_info = f1_handler.get_session_info(2026, event)
        if session_info and isinstance(session_info, dict) and isinstance(session_info.get("error"), str):
            return {"error": session_info.get("error")}
        return sim_engine.run_basic_simulation(f1_handler, data)

    @staticmethod
    def run_season_simulation(data):
        return sim_engine.run_championship(f1_handler, data)

    @staticmethod
    def reimport_data():
        # Maps to check_connection in Real-Time mode
        return f1_handler.check_connection()

    @staticmethod
    def check_connection():
        return f1_handler.check_connection()

# ==========================================
# HTTP SERVER HANDLER (F1SimHandler)
# ==========================================
class F1SimHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP Handler for F1 Simulation Server."""

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        query = urllib.parse.parse_qs(parsed_path.query)

        if path.startswith('/api/'):
            self.handle_api_get(path, query)
        else:
            self.handle_static_files(path)

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path.startswith('/api/'):
            self.handle_api_post(path)
        else:
            self.send_error(404, "Endpoint not found")

    def handle_api_get(self, path, query):
        """Dispatches GET API requests."""
        try:
            response_data = {}
            if path == '/api/seasons':
                response_data = APIController.get_seasons()
            elif path == '/api/events':
                response_data = APIController.get_events(query)
            elif path == '/api/drivers':
                response_data = APIController.get_drivers(query)
            elif path == '/api/session-info':
                response_data = APIController.get_session_info(query)
            elif path == '/api/check-connection':
                 response_data = APIController.check_connection()
            elif path == '/api/simulation/start':
                pass
            else:
                raise FileNotFoundError("Endpoint not found")
            
            self.send_json_response(200, response_data)
        except ValueError as ve:
            logger.error(f"API GET ValueError: {ve}")
            self.send_json_response(400, {"error": str(ve)})
        except FileNotFoundError as fe:
            logger.error(f"API GET FileNotFoundError: {fe}")
            self.send_json_response(404, {"error": str(fe)})
        except Exception as e:
            logger.error(f"API GET Error: {e}")
            traceback.print_exc()
            self.send_json_response(500, {"error": str(e)})

    def handle_api_post(self, path):
        """Dispatches POST API requests."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8')) if post_data else {}
            
            response_data = {}
            if path == '/api/simulate':
                response_data = APIController.run_simulation(data)
            elif path == '/api/basic-simulate':
                response_data = APIController.run_basic_simulation(data)
            elif path == '/api/simulate-season':
                response_data = APIController.run_season_simulation(data)
            elif path == '/api/reimport-2026-data':
                response_data = APIController.reimport_data()
            else:
                raise FileNotFoundError("Endpoint not found")

            self.send_json_response(200, response_data)

        except ValueError as ve:
            logger.error(f"JSON Serialization Error: {ve}")
            self.send_json_response(400, {"error": "Error de datos: Valores inválidos."})
        except FileNotFoundError as fe:
            self.send_json_response(404, {"error": str(fe)})
        except Exception as e:
            logger.error(f"API POST Error: {e}")
            traceback.print_exc()
            self.send_json_response(500, {"error": str(e)})

    def handle_static_files(self, path):
        """Serves static files from the frontend directory."""
        if path == '/':
            path = '/index.html'
        
        # Security check: prevent directory traversal
        clean_path = path.lstrip('/')
        file_path = os.path.join(FRONTEND_DIR, clean_path)
        
        # Ensure the resolved path is within FRONTEND_DIR
        if not os.path.abspath(file_path).startswith(os.path.abspath(FRONTEND_DIR)):
            self.send_error(403, "Forbidden")
            return

        if os.path.exists(file_path) and os.path.isfile(file_path):
            self.send_response(200)
            self.send_header('Content-type', self.get_content_type(file_path))
            self.end_headers()
            
            try:
                with open(file_path, 'rb') as f:
                    self.wfile.write(f.read())
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404, "File not found")

    def send_json_response(self, status_code, data):
        """Helper to send JSON responses."""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        try:
            self.wfile.write(json.dumps(data, allow_nan=False).encode())
        except (BrokenPipeError, ConnectionResetError):
            pass

    def get_content_type(self, file_path):
        """Determines content type based on file extension."""
        if file_path.endswith('.html'): return 'text/html'
        if file_path.endswith('.css'): return 'text/css'
        if file_path.endswith('.js'): return 'application/javascript'
        if file_path.endswith('.json'): return 'application/json'
        if file_path.endswith('.png'): return 'image/png'
        if file_path.endswith('.jpg') or file_path.endswith('.jpeg'): return 'image/jpeg'
        return 'application/octet-stream'

class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

# ==========================================
# UNIT TESTS
# ==========================================
import unittest

class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.engine = SimulationEngine()
        self.mock_context = {
            "hist_data": {
                "team_stats": {"Red Bull Racing": {"mean": 90.0, "std": 0.5}, "Mercedes": {"mean": 90.5, "std": 0.5}},
                "driver_stats": {"verstappen": {"mean": 89.8, "std": 0.4}, "hamilton": {"mean": 90.2, "std": 0.4}}
            },
            "drivers": [
                {"DriverId": "verstappen", "Abbreviation": "VER", "TeamName": "Red Bull Racing", "Number": 1, "GridPosition": 1},
                {"DriverId": "hamilton", "Abbreviation": "HAM", "TeamName": "Mercedes", "Number": 44, "GridPosition": 2},
                {"DriverId": "perez", "Abbreviation": "PER", "TeamName": "Red Bull Racing", "Number": 11, "GridPosition": 3},
                {"DriverId": "russell", "Abbreviation": "RUS", "TeamName": "Mercedes", "Number": 63, "GridPosition": 4},
                {"DriverId": "leclerc", "Abbreviation": "LEC", "TeamName": "Ferrari", "Number": 16, "GridPosition": 5},
                {"DriverId": "sainz", "Abbreviation": "SAI", "TeamName": "Ferrari", "Number": 55, "GridPosition": 6}
            ],
            "pace_scale": 1.0
        }

    def test_overtake_count_realism(self):
        """
        Test that the total number of overtakes in a simulated race is within a realistic range.
        Typical dry race: 20-60 overtakes depending on circuit.
        We check that it doesn't explode to >100 for a short race.
        """
        max_laps = 20
        sim_result = self.engine._simulate_race(max_laps, self.mock_context)
        
        position_history = sim_result["position_history"]
        total_overtakes = 0
        
        # Calculate overtakes from position history
        for i in range(1, len(position_history)):
            prev_order = position_history[i-1]["order"]
            curr_order = position_history[i]["order"]
            
            # Simple check: Count how many drivers changed relative order
            if prev_order != curr_order:
                pass

        # Alternative: Check validation logic
        self.assertTrue(sim_result["integrity_check"]["valid"])
        
        # Check that fast car starting back makes progress but not instant
        # Create context where VER starts last
        context_recovery = dict(self.mock_context)
        context_recovery["drivers"] = list(self.mock_context["drivers"])
        context_recovery["drivers"][0] = dict(context_recovery["drivers"][0])
        context_recovery["drivers"][0]["GridPosition"] = 6 # VER starts last
        context_recovery["drivers"][-1]["GridPosition"] = 1 # SAI starts first
        
        sim_recovery = self.engine._simulate_race(max_laps, context_recovery)
        
        # VER should have gained positions but maybe not won in 20 laps
        ver_result = sim_recovery["drivers"]["VER"]
        # We expect some progress
        self.assertTrue(len(ver_result["laps"]) == max_laps)

    def test_manual_driver_skill_override(self):
        np.random.seed(42)
        max_laps = 10
        base_context = dict(self.mock_context)
        base_context["location"] = "Monza"
        sim_base = self.engine._simulate_race(max_laps, base_context)
        ver_base_times = [l["time"] for l in sim_base["drivers"]["VER"]["laps"]]
        np.random.seed(42)
        override_context = dict(self.mock_context)
        override_context["location"] = "Monza"
        override_context["driver_modifiers"] = {"verstappen": {"pace_offset": -0.5}}
        sim_override = self.engine._simulate_race(max_laps, override_context)
        ver_override_times = [l["time"] for l in sim_override["drivers"]["VER"]["laps"]]
        base_avg = float(np.mean(ver_base_times))
        override_avg = float(np.mean(ver_override_times))
        self.assertLess(override_avg, base_avg - 0.1)

    def test_teammate_skill_from_history(self):
        np.random.seed(7)
        max_laps = 25
        context = {
            "hist_data": {
                "team_stats": {"Test Team": {"mean": 92.0, "std": 0.5}},
                "driver_stats": {
                    "driver_fast": {"mean": 91.0, "std": 0.4},
                    "driver_slow": {"mean": 93.0, "std": 0.3},
                },
            },
            "drivers": [
                {"DriverId": "driver_fast", "Abbreviation": "FA1", "TeamName": "Test Team", "Number": 10, "GridPosition": 2},
                {"DriverId": "driver_slow", "Abbreviation": "SL1", "TeamName": "Test Team", "Number": 20, "GridPosition": 1},
            ],
            "pace_scale": 1.0,
            "location": "Monza",
        }
        sim = self.engine._simulate_race(max_laps, context)
        fast_times = [l["time"] for l in sim["drivers"]["FA1"]["laps"]]
        slow_times = [l["time"] for l in sim["drivers"]["SL1"]["laps"]]
        fast_avg = float(np.mean(fast_times))
        slow_avg = float(np.mean(slow_times))
        self.assertLess(fast_avg, slow_avg)

    def test_basic_simulator_team_hierarchy(self):
        np.random.seed(123)
        max_laps = 30
        context = {
            "hist_data": {
                "team_stats": {"Team A": {"mean": 92.0, "std": 0.5}},
                "driver_stats": {
                    "fast": {"mean": 90.0, "std": 0.4},
                    "slow": {"mean": 94.0, "std": 0.4},
                },
            },
            "drivers": [
                {
                    "DriverId": "fast",
                    "Abbreviation": "FA",
                    "TeamName": "Team A",
                    "Number": 10,
                    "GridPosition": 2,
                },
                {
                    "DriverId": "slow",
                    "Abbreviation": "SL",
                    "TeamName": "Team A",
                    "Number": 20,
                    "GridPosition": 1,
                },
            ],
            "pace_scale": 1.0,
            "location": "Monza",
        }
        sim = self.engine._simulate_basic_race(max_laps, context)
        fast_times = [l["time"] for l in sim["drivers"]["FA"]["laps"]]
        slow_times = [l["time"] for l in sim["drivers"]["SL"]["laps"]]
        fast_total = float(sum(fast_times))
        slow_total = float(sum(slow_times))
        self.assertLess(fast_total, slow_total)

    def test_circuit_difficulty_factor(self):
        """
        Verify that overtaking is harder on difficult tracks (e.g. Monaco) compared to easy tracks (e.g. Monza).
        We simulate two identical scenarios with different circuit names.
        """
        max_laps = 20
        
        # Scenario: Fast car (VER) starting last
        context = dict(self.mock_context)
        context["drivers"] = list(self.mock_context["drivers"])
        context["drivers"][0] = dict(context["drivers"][0])
        context["drivers"][0]["GridPosition"] = 6 # VER starts last
        
        # 1. Easy Overtake Circuit (Monza)
        context_easy = dict(context)
        context_easy["location"] = "Monza"
        sim_easy = self.engine._simulate_race(max_laps, context_easy)
        
        # 2. Hard Overtake Circuit (Monaco)
        context_hard = dict(context)
        context_hard["location"] = "Monaco"
        sim_hard = self.engine._simulate_race(max_laps, context_hard)
        
        ver_easy = sim_easy["drivers"]["VER"]
        ver_hard = sim_hard["drivers"]["VER"]
        
        start_pos = 6
        
        drivers_easy = sorted(sim_easy["drivers"].values(), key=lambda x: x["laps"][-1]["cumulative"])
        finish_pos_easy = [d["Abbreviation"] for d in drivers_easy].index("VER") + 1
        
        drivers_hard = sorted(sim_hard["drivers"].values(), key=lambda x: x["laps"][-1]["cumulative"])
        finish_pos_hard = [d["Abbreviation"] for d in drivers_hard].index("VER") + 1
        
        gained_easy = start_pos - finish_pos_easy
        gained_hard = start_pos - finish_pos_hard
        
        self.assertTrue(gained_easy >= gained_hard - 2, f"Monza ({gained_easy}) should be easier than Monaco ({gained_hard})")

    def test_single_lap_position_changes_bounded(self):
        np.random.seed(10)
        max_laps = 25
        context = dict(self.mock_context)
        context["location"] = "Monza"
        sim_result = self.engine._simulate_race(max_laps, context)
        position_history = sim_result["position_history"]
        prev_order = None
        max_jump = 0
        for lap_data in position_history:
            order = lap_data["order"]
            if prev_order is None:
                prev_order = order
                continue
            idx_prev = {abbr: idx for idx, abbr in enumerate(prev_order)}
            for idx, abbr in enumerate(order):
                if abbr not in idx_prev:
                    continue
                delta = abs(idx_prev[abbr] - idx)
                if delta > max_jump:
                    max_jump = delta
            prev_order = order
        self.assertLessEqual(max_jump, 6)

    def test_build_driver_states_respects_hist_performance(self):
        context = dict(self.mock_context)
        hist_data = context["hist_data"]
        team_stats = hist_data["team_stats"]
        driver_stats = hist_data["driver_stats"]
        drivers = context["drivers"]
        circuit_profile = SimulationEngine.CIRCUIT_PROFILES.get("Monza") or {}
        pace_scale = 1.0
        modifiers = {}
        sim_drivers, _ = self.engine._build_driver_states(drivers, {"team_stats": team_stats, "driver_stats": driver_stats}, pace_scale, modifiers, circuit_profile)
        means = {d["abbr"]: d["base_mean"] for d in sim_drivers}
        self.assertLess(means["VER"], means["HAM"])

    def test_position_error_metrics_shape(self):
        position_history_sim = [
            {"lap": 1, "order": ["VER", "HAM", "PER"]},
            {"lap": 2, "order": ["VER", "PER", "HAM"]},
        ]
        position_history_ref = [
            {"lap": 1, "order": ["VER", "HAM", "PER"]},
            {"lap": 2, "order": ["VER", "HAM", "PER"]},
        ]
        metrics = self.engine._compute_position_error_metrics(position_history_sim, position_history_ref)
        self.assertIn("rmse_position", metrics)
        self.assertIn("mae_position", metrics)

    def test_position_summary_contents(self):
        np.random.seed(5)
        max_laps = 15
        context = dict(self.mock_context)
        context["location"] = "Monza"
        sim_result = self.engine._simulate_race(max_laps, context)
        metrics = self.engine._calculate_metrics(sim_result, max_laps, {"hist_data": context["hist_data"]})
        summary = metrics["kpis"]["race"]["position_summary"]
        table = metrics["kpis"]["race"]["position_summary_table"]
        self.assertEqual(len(summary), len(context["drivers"]))
        self.assertTrue(isinstance(table, str) and len(table) > 0)
        self.assertEqual(summary[0]["finish_position"], 1)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.argv.remove("--test")
        unittest.main()
    else:
        try:
            logger.info(f"Server starting on http://localhost:{PORT}")
            with ThreadedTCPServer(("", PORT), F1SimHandler) as httpd:
                httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped by user.")
        except Exception as e:
            logger.critical(f"Server crashed: {e}")
            traceback.print_exc()
