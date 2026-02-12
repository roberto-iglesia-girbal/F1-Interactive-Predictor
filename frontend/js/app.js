/**
 * F1 Race Simulator - Frontend Application
 * Refactored and Optimized for Modular Architecture
 * 
 * SIMPLIFIED CONFIGURATION MODE:
 * Only Circuit Selection is available to the user.
 * Default parameters: Pace Scale=1.0, Active Filter=Enabled, Season=2026.
 */

// ==========================================
// 1. Classes & Helpers
// ==========================================

class PrecisePlaybackClock {
    constructor(options) {
        const opts = options || {};
        this.time = Number.isFinite(opts.time) ? opts.time : 0;
        this.minTime = Number.isFinite(opts.minTime) ? opts.minTime : 0;
        this.maxTime = Number.isFinite(opts.maxTime) ? opts.maxTime : 0;
        this.timeScale = Number.isFinite(opts.timeScale) ? opts.timeScale : 10;
        this.forwardSpeed = Number.isFinite(opts.forwardSpeed) ? opts.forwardSpeed : 1.0;
        this.reverseSpeed = Number.isFinite(opts.reverseSpeed) ? opts.reverseSpeed : 1.0;
        this.direction = opts.direction === -1 ? -1 : 1;
        this.running = !!opts.running;
        this._lastNow = null;
    }

    setBounds(minTime, maxTime) {
        if (Number.isFinite(minTime)) this.minTime = minTime;
        if (Number.isFinite(maxTime)) this.maxTime = maxTime;
        this.time = this._clamp(this.time);
    }

    setSpeeds(forwardSpeed, reverseSpeed) {
        if (Number.isFinite(forwardSpeed) && forwardSpeed > 0) this.forwardSpeed = forwardSpeed;
        if (Number.isFinite(reverseSpeed) && reverseSpeed > 0) this.reverseSpeed = reverseSpeed;
    }

    setDirection(direction, now) {
        const next = direction === -1 ? -1 : 1;
        if (this.direction !== next) {
            this.direction = next;
            if (Number.isFinite(now)) this._lastNow = now;
        }
    }

    setRunning(running, now) {
        const next = !!running;
        if (this.running !== next) {
            this.running = next;
            if (Number.isFinite(now)) this._lastNow = now;
        }
    }

    setTime(time, now) {
        const t = Number(time);
        this.time = this._clamp(Number.isFinite(t) ? t : this.time);
        if (Number.isFinite(now)) this._lastNow = now;
    }

    tick(now) {
        if (!Number.isFinite(now)) return { time: this.time, delta: 0 };
        if (this._lastNow == null) {
            this._lastNow = now;
            return { time: this.time, delta: 0 };
        }
        const dt = (now - this._lastNow) / 1000;
        this._lastNow = now;

        if (!this.running) return { time: this.time, delta: 0 };
        if (!Number.isFinite(dt) || dt <= 0) return { time: this.time, delta: 0 };

        const rate = this.direction === -1 ? this.reverseSpeed : this.forwardSpeed;
        const delta = dt * rate * this.timeScale * this.direction;
        const next = this._clamp(this.time + delta);
        const applied = next - this.time;
        this.time = next;

        if ((this.time <= this.minTime && this.direction === -1) || (this.maxTime > 0 && this.time >= this.maxTime && this.direction === 1)) {
            this.running = false;
        }

        return { time: this.time, delta: applied };
    }

    syncNow(now) {
        if (Number.isFinite(now)) this._lastNow = now;
    }

    _clamp(t) {
        let out = t;
        if (Number.isFinite(this.minTime)) out = Math.max(this.minTime, out);
        if (Number.isFinite(this.maxTime) && this.maxTime > 0) out = Math.min(this.maxTime, out);
        return out;
    }
}

// ==========================================
// 2. Main Application Module
// ==========================================

const F1Sim = {
    // --- Constants ---
    Constants: {
        API_URL: '/api',
        LEADERBOARD_ROW_HEIGHT: 44,
        COMPOUND_COLORS: {
            'SOFT': '#ff3b30',
            'MEDIUM': '#ffcc00',
            'HARD': '#ffffff',
            'INTERMEDIATE': '#34c759',
            'WET': '#007aff'
        },
        SECTOR_COLORS: ['#ff3b30', '#007aff', '#ffcc00'],
        VISUAL_CONFIG: {
            tecnologia_renderizado: 'Canvas2D',
            paletas: {
                default: {
                    sectores: ['#ff3b30', '#007aff', '#ffcc00'],
                    pista_base: '#2a2a2a',
                    linea_central: '#d0d0d0'
                },
                daltonismo: {
                    sectores: ['#E69F00', '#56B4E9', '#009E73'],
                    pista_base: '#2a2a2a',
                    linea_central: '#d0d0d0'
                }
            },
            dorsal: {
                tam_min_px: 24,
                tipo_letra: "'Titillium Web', Arial, sans-serif"
            },
            coche: {
                radio_px: 18,
                borde: '#0b0b0b',
                borde_ancho_px: 2
            }
        }
    },

    // --- State Management ---
    State: {
        config: {
            season: null,
            event: null,
            drivers: [],
            gridOrder: [],
            officialLaps: null,
            eventMeta: null,
            activeDriversFilterEnabled: true
        },
        simData: null,
        simState: {
            running: false,
            time: 0,
            speed: 1.0,
            speedTarget: 1.0,
            reverseSpeed: 1.0,
            direction: 1,
            minTime: 0,
            maxTime: 0,
            timeScale: 1.0,
            clock: null,
            lastUiNow: null,
            scrubbing: false,
            finished: false,
            _resumeAfterVisibility: false,
            _speedBeforePause: 1.0,
            _lastLoopNowMs: null,
            _lastSimTimeWhileRunning: null,
            positions: [],
            lastPositions: {},
            viewMode: 'physical',
            rotation: 0,
            zoom: 1.0,
            panX: 0,
            panY: 0,
            isDragging: false,
            lastMouseX: 0,
            lastMouseY: 0,
            baseScale: 1,
            bounds: { centerX: 0, centerY: 0 },
            features: { corners: [], sectors: [] },
            errorState: { hasError: false, lastError: null, count: 0 },
            leaderboardPaging: { pageSize: 15, currentPage: 1, totalPages: 1 },
            leaderboardFilterLimit: null,
            userDriverId: null,
            metricsHistory: { samples: [], lastSampleTime: 0 },
            positionTracking: { perDriver: {}, totalGained: 0, totalLost: 0, events: [] },
            positionChangeRef: {},
            _restoreTime: null,
            colorMode: 'default'
        }
    },

    // --- Command Pattern Implementation ---
    Commands: {
        StartSimulation: class {
            constructor(engine) { this.engine = engine; }
            execute() { this.engine.startSimulation(); }
        },
        TogglePause: class {
            constructor(engine) { this.engine = engine; }
            execute() { this.engine.togglePause(); }
        },
        StopSimulation: class {
            constructor(engine) { this.engine = engine; }
            execute() { this.engine.stop(); }
        },
        ChangeSpeed: class {
            constructor(engine, speed) { this.engine = engine; this.speed = speed; }
            execute() { this.engine.setSpeed(this.speed); }
        },
        SetTime: class {
            constructor(engine, time, isScrubbing) { 
                this.engine = engine; 
                this.time = time; 
                this.isScrubbing = isScrubbing;
            }
            execute() { this.engine.setTime(this.time, this.isScrubbing); }
        }
    },

    Invoker: {
        history: [],
        execute(command) {
            if (command && typeof command.execute === 'function') {
                command.execute();
                this.history.push(command);
            } else {
                console.error("Invalid command", command);
            }
        }
    },

    // --- Utility Functions ---
    Utils: {
        formatMetricNumber(value, decimals) {
            const d = Number.isFinite(decimals) ? decimals : 2;
            const n = typeof value === 'string' ? Number(value) : value;
            if (!Number.isFinite(n)) return '—';
            return n.toFixed(d);
        },

        formatMetric(value, unit, decimals) {
            const text = this.formatMetricNumber(value, decimals);
            if (text === '—') return text;
            return unit ? `${text} ${unit}` : text;
        },

        formatTime(seconds) {
            if (seconds < 0) return "0:00.000";
            const m = Math.floor(seconds / 60);
            const s = Math.floor(seconds % 60);
            const ms = Math.floor((seconds % 1) * 1000);
            return `${m}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
        },

        clampSimTime(t) {
            const s = F1Sim.State.simState;
            let out = Number.isFinite(t) ? t : 0;
            out = Math.max(s.minTime || 0, out);
            if (Number.isFinite(s.maxTime) && s.maxTime > 0) out = Math.min(s.maxTime, out);
            return out;
        },

        getNowMs() {
            if (typeof performance !== 'undefined' && performance && typeof performance.now === 'function') {
                return performance.now();
            }
            return Date.now();
        },

        isOfficialEvent(ev) {
            if (!ev) return false;
            const round = Number(ev.RoundNumber);
            if (!Number.isFinite(round) || round <= 0) return false;
            const name = String(ev.OfficialEventName || ev.EventName || '').toLowerCase();
            const country = String(ev.Country || '').toLowerCase();
            const location = String(ev.Location || '').toLowerCase();
            const text = `${name} ${country} ${location}`;
            const banned = ['test', 'testing', 'pre-season', 'preseason', 'shakedown', 'practice', 'private', 'development'];
            for (let i = 0; i < banned.length; i++) {
                if (text.includes(banned[i])) return false;
            }
            return true;
        },

        getTeamColor(teamName) {
            const drivers = F1Sim.State.simData && F1Sim.State.simData.drivers;
            if (drivers) {
                for (let d in drivers) {
                    if (drivers[d].team === teamName) return drivers[d].color;
                }
            }
            return '#ffffff';
        },

        parseHexColor(input) {
            const s = String(input || '').trim();
            if (!s) return null;
            const hex = s.startsWith('#') ? s.slice(1) : s;
            if (hex.length === 3) {
                const r = parseInt(hex[0] + hex[0], 16);
                const g = parseInt(hex[1] + hex[1], 16);
                const b = parseInt(hex[2] + hex[2], 16);
                if ([r, g, b].every(Number.isFinite)) return { r, g, b };
                return null;
            }
            if (hex.length === 6) {
                const r = parseInt(hex.slice(0, 2), 16);
                const g = parseInt(hex.slice(2, 4), 16);
                const b = parseInt(hex.slice(4, 6), 16);
                if ([r, g, b].every(Number.isFinite)) return { r, g, b };
                return null;
            }
            return null;
        },

        relativeLuminance(rgb) {
            const toLinear = (c) => {
                const v = c / 255;
                return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
            };
            const r = toLinear(rgb.r);
            const g = toLinear(rgb.g);
            const b = toLinear(rgb.b);
            return 0.2126 * r + 0.7152 * g + 0.0722 * b;
        },

        contrastRatio(foreHex, backHex) {
            const fg = this.parseHexColor(foreHex);
            const bg = this.parseHexColor(backHex);
            if (!fg || !bg) return 1;
            const L1 = this.relativeLuminance(fg);
            const L2 = this.relativeLuminance(bg);
            const lighter = Math.max(L1, L2);
            const darker = Math.min(L1, L2);
            return (lighter + 0.05) / (darker + 0.05);
        },

        pickReadableTextColor(bgHex) {
            const white = '#ffffff';
            const black = '#000000';
            const cWhite = this.contrastRatio(white, bgHex);
            const cBlack = this.contrastRatio(black, bgHex);
            if (cWhite >= cBlack) return { color: white, contrast: cWhite };
            return { color: black, contrast: cBlack };
        },

        setColorMode(mode) {
            const next = mode === 'daltonismo' ? 'daltonismo' : 'default';
            F1Sim.State.simState.colorMode = next;
            try {
                localStorage.setItem('f1sim_color_mode', next);
            } catch (e) {
            }
            F1Sim.Renderer.draw();
        },

        validateState() {
            const s = F1Sim.State.simState;
            const errors = [];
            
            if (!Number.isFinite(s.time)) errors.push("Time is NaN");
            
            if (s.metricsHistory && s.metricsHistory.lastSampleTime > s.time + 2.0) {
                 // Allow margin for discrete sampling/intervals
                errors.push(`Metrics future leak: lastSample ${s.metricsHistory.lastSampleTime} > time ${s.time}`);
            }
            
            if (errors.length > 0) {
                console.warn("State Validation Errors:", errors);
                return false;
            }
            return true;
        }
    },

    // --- API Interactions ---
    API: {
        async fetchSeasons() {
            try {
                const res = await fetch(`${F1Sim.Constants.API_URL}/seasons`);
                return await res.json();
            } catch (err) {
                console.error("Error loading seasons", err);
                throw err;
            }
        },

        async fetchEvents(year, activeOnly) {
            try {
                const url = new URL(`${F1Sim.Constants.API_URL}/events`);
                if (year) {
                    url.searchParams.set('year', year);
                }
                if (activeOnly) {
                    url.searchParams.set('active_only', '1');
                }
                const res = await fetch(url.toString());
                if (!res.ok) throw new Error(`Error HTTP: ${res.status}`);
                return await res.json();
            } catch (err) {
                console.error("Error loading events", err);
                throw err;
            }
        },

        async fetchSessionInfo(eventId) {
            try {
                const res = await fetch(`${F1Sim.Constants.API_URL}/session-info?event_id=${eventId}`);
                return await res.json();
            } catch (err) {
                console.error("Error loading session info", err);
                throw err;
            }
        },

        async fetchDrivers(eventId) {
            try {
                const url = new URL(`${F1Sim.Constants.API_URL}/drivers`);
                url.searchParams.set('event_id', eventId);
                const res = await fetch(url.toString());
                return await res.json();
            } catch (err) {
                console.error("Error loading drivers", err);
                throw err;
            }
        },

        async runSimulation(payload) {
            try {
                const res = await fetch(`${F1Sim.Constants.API_URL}/simulate`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                return await res.json();
            } catch (err) {
                console.error("Error starting simulation", err);
                throw err;
            }
        },

        async checkConnection() {
            try {
                const res = await fetch(`${F1Sim.Constants.API_URL}/check-connection`);
                return await res.json();
            } catch (err) {
                return { success: false, error: err.message };
            }
        }
    },

    // --- Tests Module ---
    Tests: {
        async runRewindTest() {
            console.log("Starting Rewind Test...");
            const s = F1Sim.State.simState;
            
            if (!s.running && s.time <= 0) {
                console.warn("Test usually requires running simulation. Starting one if ready...");
            }
            
            const t1 = s.time;
            const snap1 = JSON.stringify(s.positions);
            console.log(`Test: Captured state at T=${t1.toFixed(2)}`);
            
            const targetT = t1 + 5.0;
            console.log(`Test: Advancing to T=${targetT.toFixed(2)}...`);
            F1Sim.Engine.setTime(targetT, false);
            
            console.log("Test: Rewinding...");
            F1Sim.Engine.setTime(t1, false);
            
            const snap2 = JSON.stringify(s.positions);
            const valid = F1Sim.Utils.validateState();
            
            console.log(`Test: Rewind Complete. Time: ${s.time.toFixed(2)} (Target: ${t1.toFixed(2)})`);
            console.log(`Test: Validation Passed: ${valid}`);
            
            if (Math.abs(s.time - t1) < 0.1 && valid && snap1 === snap2) {
                console.log("%cTEST PASSED: Time restored and state valid.", "color: green; font-weight: bold;");
            } else {
                console.error("TEST FAILED: Time mismatch, state mismatch or validation error.");
            }
        },

        async runScrubDuringPlaybackTest() {
            const s = F1Sim.State.simState;
            if (!s.clock) {
                console.error("No clock. Start a simulation first.");
                return false;
            }
            if (!F1Sim.State.simData) {
                console.error("No simData. Start a simulation first.");
                return false;
            }
            if (s.finished) {
                console.error("Simulation finished. Start a new simulation first.");
                return false;
            }

            const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

            F1Sim.Engine.setSpeed(2.0);
            const t0 = s.time;
            await sleep(200);
            const t0b = s.time;
            if (!(t0b > t0)) {
                console.error("FAILED: time did not advance before scrubbing.");
                return false;
            }

            const target = Math.max(s.minTime || 0, t0b - 15);
            F1Sim.Engine.setTime(target, false);
            const t1 = s.time;

            await sleep(200);
            const t1b = s.time;
            if (t1b > t0b - 5) {
                console.error("FAILED: time snapped back near pre-scrub time.");
                return false;
            }
            if (!(t1b > t1)) {
                console.error("FAILED: time did not advance after scrubbing.");
                return false;
            }

            const ok = F1Sim.Utils.validateState();
            if (!ok) {
                console.error("FAILED: validation errors after scrubbing.");
                return false;
            }

            console.log("%cTEST PASSED: Scrub back and resume playback OK.", "color: green; font-weight: bold;");
            return true;
        },

        runCircuitDropdownFilterTest() {
            const events = [
                { RoundNumber: 1, OfficialEventName: 'Bahrain Grand Prix', Country: 'Bahrain', Location: 'Sakhir' },
                { RoundNumber: 2, OfficialEventName: 'Saudi Arabian Grand Prix', Country: 'Saudi Arabia', Location: 'Jeddah' },
                { RoundNumber: 3, OfficialEventName: 'Australian Grand Prix', Country: 'Australia', Location: 'Melbourne' }
            ];
            F1Sim.State.config.availableEvents = events;
            if (F1Sim.UI.buildEventDropdown) {
                F1Sim.UI.buildEventDropdown(events);
            }
            const input = F1Sim.UI.elements['event-select-input'];
            if (!input) {
                console.error("Dropdown input not found");
                return false;
            }
            input.value = 'bahrain';
            F1Sim.UI.filterEventDropdown(input.value);
            const dropdown = F1Sim.UI.elements['event-select-dropdown'];
            const options = dropdown ? dropdown.querySelectorAll('.circuit-option') : [];
            const passed = options.length === 1 && options[0].textContent.toLowerCase().includes('bahrain');
            if (passed) {
                console.log("%cTEST PASSED: Circuit dropdown filter works.", "color: green; font-weight: bold;");
            } else {
                console.error("TEST FAILED: Circuit dropdown filter mismatch.");
            }
            return passed;
        }
    },

    // --- UI Controller ---
    UI: {
        elements: {},

        cacheElements() {
            const ids = [
                'event-select', 'event-select-input', 'event-select-dropdown', 'event-select-combobox', 'event-select-toggle',
                'start-sim-btn', 'loading-indicator', 'connection-status',
                'config-panel', 'simulation-panel', 'timer', 'current-lap',
                'total-laps', 'leaderboard', 'race-canvas', 'time-scrubber',
                'scrubber-tooltip', 'speed-select', 'speed-val', 'timer-remaining',
                'tooltip', 'rotation-slider', 'race-title', 'race-subtitle', 
                'play-pause-btn', 'new-sim-btn', 
                'metrics-modal', 'metrics-content', 'close-metrics-btn', 
                'zoom-in', 'zoom-out', 'zoom-reset', 'sim-error-indicator',
                'data-panel', 'grid-preview-container', 'grid-list', 'sim-status-indicator',
                'race-summary'
            ];
            ids.forEach(id => {
                this.elements[id] = document.getElementById(id);
            });
            this.ctx = this.elements['race-canvas'] ? this.elements['race-canvas'].getContext('2d') : null;

            // Bind Leaderboard Headers
            const lbHeaders = document.querySelectorAll('.leaderboard-header > div[data-sort]');
            lbHeaders.forEach(header => {
                header.addEventListener('click', () => {
                    const sortKey = header.getAttribute('data-sort');
                    const currentKey = F1Sim.State.simState.leaderboardSort;
                    
                    if (currentKey === sortKey) {
                        // Toggle direction
                        F1Sim.State.simState.leaderboardSortDir *= -1;
                    } else {
                        // New key
                        F1Sim.State.simState.leaderboardSort = sortKey;
                        // Default directions:
                        // 'change' -> Descending (Best gainers first)
                        // Others -> Ascending (Pos 1, A-Z)
                        if (sortKey === 'change') {
                            F1Sim.State.simState.leaderboardSortDir = -1;
                        } else {
                            F1Sim.State.simState.leaderboardSortDir = 1;
                        }
                    }
                    
                    // Trigger update immediately
                    F1Sim.UI.updateLeaderboard();
                });
            });
        },

        renderGridPreview(drivers) {
            const container = this.elements['grid-preview-container'];
            const list = this.elements['grid-list'];
            
            if (!container || !list) return;
    
            if (!drivers || drivers.length === 0) {
                container.classList.add('hidden');
                return;
            }
    
            container.classList.remove('hidden');
            list.innerHTML = '';
    
            const sorted = [...drivers].sort((a, b) => (a.GridPosition || 999) - (b.GridPosition || 999));
    
            sorted.forEach(d => {
                const row = document.createElement('div');
                row.className = 'driver-row';
                row.style.display = 'flex';
                row.style.alignItems = 'center';
                row.style.padding = '4px';
                row.style.borderBottom = '1px solid #eee';
    
                const pos = document.createElement('span');
                pos.textContent = (d.GridPosition || '-') + '.';
                pos.style.fontWeight = 'bold';
                pos.style.width = '30px';
                pos.style.textAlign = 'right';
                pos.style.marginRight = '10px';
    
                const colorBox = document.createElement('div');
                colorBox.style.width = '4px';
                colorBox.style.height = '20px';
                colorBox.style.backgroundColor = '#' + (d.TeamColor || '000000');
                colorBox.style.marginRight = '10px';
    
                const name = document.createElement('span');
                name.textContent = `${d.BroadcastName || d.Abbreviation}`;
                name.style.flex = '1';
    
                const team = document.createElement('span');
                team.textContent = d.TeamName || '';
                team.style.fontSize = '0.8rem';
                team.style.color = '#666';
    
                row.appendChild(pos);
                row.appendChild(colorBox);
                row.appendChild(name);
                row.appendChild(team);
                list.appendChild(row);
            });
        },

        setVisualError(message) {
            const indicator = this.elements['sim-error-indicator'];
            if (!indicator) return;
            if (typeof message === 'string' && message.trim()) {
                indicator.textContent = message;
                indicator.classList.remove('hidden');
            } else {
                indicator.classList.add('hidden');
            }
        },

        setLoadingState(isLoading, message = "Cargando...") {
            const spinner = this.elements['loading-indicator'];
            const status = this.elements['connection-status'];
            const select = this.elements['event-select'];
            const input = this.elements['event-select-input'];
            const toggle = this.elements['event-select-toggle'];
            const btn = this.elements['start-sim-btn'];

            if (isLoading) {
                if (spinner) spinner.classList.remove('hidden');
                if (status) {
                    status.textContent = message;
                    status.style.color = '#aaa';
                }
                if (select) select.disabled = true;
                if (input) {
                    input.disabled = true;
                    input.value = "";
                    input.placeholder = "Cargando calendario 2026...";
                }
                if (toggle) toggle.disabled = true;
                if (btn) btn.disabled = true;
            } else {
                if (spinner) spinner.classList.add('hidden');
                if (status) status.textContent = "";
                if (select) select.disabled = false;
                 if (input) {
                    input.disabled = false;
                    input.placeholder = "Buscar Gran Premio...";
                 }
                 if (toggle) toggle.disabled = false;
                // Btn remains disabled unless event selected
            }
        },

        resizeCanvas() {
            const canvas = this.elements['race-canvas'];
            if (!canvas) return;

            const rect = canvas.getBoundingClientRect();
            let w = Math.floor(rect.width);
            let h = Math.floor(rect.height);

            if (!(w > 0 && h > 0)) {
                const container = document.getElementById('track-container');
                if (container) {
                    const crect = container.getBoundingClientRect();
                    w = Math.floor(crect.width);
                    h = Math.floor(crect.height);
                }
            }

            if (!(w > 0 && h > 0)) {
                w = 800;
                h = 600;
            }

            if (canvas.width !== w) canvas.width = w;
            if (canvas.height !== h) canvas.height = h;
        },

        async init() {
            this.cacheElements();
            
            // Check connection first
            this.setLoadingState(true, "Verificando conexión con FastF1...");
            const conn = await F1Sim.API.checkConnection();
            
            if (conn && conn.success) {
                this.setLoadingState(true, "Conexión establecida. Obteniendo calendario...");
                await this.loadEvents();
            } else {
                const status = this.elements['connection-status'];
                if (status) {
                    status.textContent = "Error de conexión: " + (conn.error || "Desconocido");
                    status.style.color = "#ff5252";
                }
                const spinner = this.elements['loading-indicator'];
                if (spinner) spinner.classList.add('hidden');
                
                // Allow retry via reload or simple button if we added one
                const select = this.elements['event-select'];
                if (select) {
                    select.innerHTML = '<option value="">Error de conexión</option><option value="retry">🔄 Reintentar</option>';
                    select.disabled = false;
                }
            }
            
            this.setupEventListeners();
            this.setupCanvasInteractions();

            window.addEventListener('error', (e) => {
                const msg = (e && e.message) ? String(e.message) : 'Error en el frontend';
                this.setVisualError(msg);
            });
            window.addEventListener('unhandledrejection', (e) => {
                const msg = (e && e.reason) ? String(e.reason) : 'Promesa rechazada sin capturar';
                this.setVisualError(msg);
            });
            document.addEventListener('visibilitychange', () => {
                const s = F1Sim.State.simState;
                if (!s.clock) return;
                const now = F1Sim.Utils.getNowMs();

                if (document.hidden) {
                    s._resumeAfterVisibility = !!s.running;
                    if (s.running) {
                        s.clock.setRunning(false, now);
                        s.running = false;
                        F1Sim.UI.updateUI();
                    } else {
                        s.clock.syncNow(now);
                    }
                } else {
                    s.clock.syncNow(now);
                    if (s._resumeAfterVisibility && !s.finished) {
                        s._resumeAfterVisibility = false;
                        if (!(Number.isFinite(s.speedTarget) && s.speedTarget > 0)) {
                            const restore = Number.isFinite(s._speedBeforePause) && s._speedBeforePause > 0 ? s._speedBeforePause : 1.0;
                            F1Sim.Engine.setSpeed(restore);
                        } else {
                            s.clock.setRunning(true, now);
                            s.running = true;
                            F1Sim.Engine.loop();
                        }
                        F1Sim.UI.updateUI();
                    }
                }
            });
            window.addEventListener('resize', () => {
                this.resizeCanvas();
                F1Sim.Renderer.draw();
            });
        },

        async loadEvents(force = false) {
            const eventSelect = this.elements['event-select'];
            const input = this.elements['event-select-input'];
            const dropdown = this.elements['event-select-dropdown'];
            if (!eventSelect) return;

            // Force 2026 Season
            const year = 2026;
            F1Sim.State.config.season = year;
            F1Sim.State.config.activeDriversFilterEnabled = false;

            this.setLoadingState(true, `Obteniendo calendario 2026...`);

            try {
                const events = await F1Sim.API.fetchEvents(year, false);
                
                if (!Array.isArray(events)) {
                    throw new Error(events.error || "Respuesta inválida");
                }

                const officialEvents = events
                    .filter(F1Sim.Utils.isOfficialEvent)
                    .sort((a, b) => {
                        const da = a.EventDate ? String(a.EventDate) : "";
                        const db = b.EventDate ? String(b.EventDate) : "";
                        if (da && db && da !== db) {
                            return da < db ? -1 : 1;
                        }
                        const ra = Number(a.RoundNumber) || 0;
                        const rb = Number(b.RoundNumber) || 0;
                        return ra - rb;
                    });

                eventSelect.innerHTML = '';

                const seenRounds = new Set();
                let addedCount = 0;

                officialEvents.forEach(ev => {
                    const r = Number(ev.RoundNumber);
                    if (!Number.isFinite(r) || r <= 0) return;
                    const key = String(r);
                    if (seenRounds.has(key)) return;
                    seenRounds.add(key);

                    const opt = document.createElement('option');
                    opt.value = ev.RoundNumber;
                    opt.dataset.eventName = String(ev.OfficialEventName || '');
                    opt.dataset.country = String(ev.Country || '');
                    opt.dataset.location = String(ev.Location || '');
                    
                    opt.textContent = `${ev.RoundNumber}. ${ev.OfficialEventName} - ${ev.Country}`;
                    eventSelect.appendChild(opt);
                    addedCount++;
                });

                if (addedCount > 0) {
                    eventSelect.size = addedCount;
                }

                if (addedCount === 0) {
                    eventSelect.innerHTML = '<option value="">No hay circuitos disponibles</option><option value="retry">🔄 Recargar</option>';
                    if (input) {
                        input.value = "";
                        input.placeholder = "No hay circuitos disponibles";
                    }
                } else if (input) {
                    input.value = "";
                    input.placeholder = "Buscar Gran Premio...";
                }
                F1Sim.State.config.availableEvents = officialEvents;
                if (dropdown) {
                    this.buildEventDropdown(officialEvents);
                }
                this.setLoadingState(false);
            } catch (err) {
                console.error("Error loading events:", err);
                eventSelect.innerHTML = `<option value="">Error: ${err.message}</option><option value="retry">🔄 Reintentar</option>`;
                this.setLoadingState(false);
                const status = this.elements['connection-status'];
                if (status) {
                    status.textContent = "Error al cargar eventos.";
                    status.style.color = "#ff5252";
                }
                if (input) {
                    input.value = "";
                    input.placeholder = "Error al cargar eventos";
                }
            }
        },

        buildEventDropdown(events) {
            const dropdown = this.elements['event-select-dropdown'];
            if (!dropdown) return;
            const list = document.createElement('div');
            list.className = 'circuit-combobox-list';
            events.forEach(ev => {
                const option = document.createElement('div');
                option.className = 'circuit-option';
                option.setAttribute('role', 'option');
                const value = String(ev.RoundNumber);
                option.dataset.value = value;
                const title = document.createElement('div');
                title.className = 'circuit-option-main';
                const t = document.createElement('div');
                t.className = 'circuit-option-title';
                t.textContent = `${ev.RoundNumber}. ${ev.OfficialEventName}`;
                const sub = document.createElement('div');
                sub.className = 'circuit-option-sub';
                const dateText = ev.EventDate ? ` • ${ev.EventDate}` : '';
                sub.textContent = `${ev.Country} • ${ev.Location}${dateText}`;
                title.appendChild(t);
                title.appendChild(sub);
                const badge = document.createElement('div');
                badge.className = 'circuit-option-badge';
                badge.textContent = value;
                option.appendChild(title);
                option.appendChild(badge);
                option.addEventListener('click', () => {
                    this.selectEventByValue(value);
                    this.closeEventDropdown();
                });
                list.appendChild(option);
            });
            if (!events.length) {
                const empty = document.createElement('div');
                empty.className = 'circuit-option-empty';
                empty.textContent = 'No hay circuitos disponibles';
                list.appendChild(empty);
            }
            dropdown.innerHTML = '';
            dropdown.appendChild(list);
            this._eventFilter = '';
            this._eventActiveIndex = 0;
            this.syncSelectedOption();
        },

        filterEventDropdown(query) {
            this._eventFilter = String(query || '').toLowerCase();
            const events = F1Sim.State.config.availableEvents || [];
            if (!this._eventFilter) {
                this.buildEventDropdown(events);
                return;
            }
            const filtered = events.filter(ev => {
                const text = `${ev.RoundNumber} ${ev.OfficialEventName} ${ev.Country} ${ev.Location}`.toLowerCase();
                return text.includes(this._eventFilter);
            });
            this.buildEventDropdown(filtered);
        },

        openEventDropdown() {
            const dropdown = this.elements['event-select-dropdown'];
            const combo = this.elements['event-select-combobox'];
            if (!dropdown || !combo) return;
            dropdown.classList.add('open');
            combo.setAttribute('aria-expanded', 'true');
        },

        closeEventDropdown() {
            const dropdown = this.elements['event-select-dropdown'];
            const combo = this.elements['event-select-combobox'];
            if (!dropdown || !combo) return;
            dropdown.classList.remove('open');
            combo.setAttribute('aria-expanded', 'false');
        },

        moveActiveOption(delta) {
            const dropdown = this.elements['event-select-dropdown'];
            if (!dropdown) return;
            const options = dropdown.querySelectorAll('.circuit-option');
            if (!options.length) return;
            if (typeof this._eventActiveIndex !== 'number') this._eventActiveIndex = 0;
            const count = options.length;
            let next = this._eventActiveIndex + delta;
            if (next < 0) next = count - 1;
            if (next >= count) next = 0;
            this._eventActiveIndex = next;
            options.forEach((opt, idx) => {
                if (idx === next) opt.classList.add('active');
                else opt.classList.remove('active');
            });
            const active = options[next];
            if (active && typeof active.scrollIntoView === 'function') {
                active.scrollIntoView({ block: 'nearest' });
            }
        },

        selectActiveOption() {
            const dropdown = this.elements['event-select-dropdown'];
            if (!dropdown) return;
            const options = dropdown.querySelectorAll('.circuit-option');
            if (!options.length) return;
            const idx = typeof this._eventActiveIndex === 'number' ? this._eventActiveIndex : 0;
            const opt = options[idx];
            if (!opt) return;
            const value = opt.dataset.value;
            if (!value) return;
            this.selectEventByValue(value);
            this.closeEventDropdown();
        },

        selectEventByValue(value) {
            const select = this.elements['event-select'];
            const input = this.elements['event-select-input'];
            if (!select) return;
            select.value = String(value);
            const ev = F1Sim.State.config.availableEvents || [];
            const match = ev.find(e => String(e.RoundNumber) === String(value));
            if (match && input) {
                input.value = `${match.RoundNumber}. ${match.OfficialEventName} - ${match.Country}`;
            }
            const event = new Event('change', { bubbles: true });
            select.dispatchEvent(event);
            this.syncSelectedOption();
        },

        syncSelectedOption() {
            const dropdown = this.elements['event-select-dropdown'];
            const select = this.elements['event-select'];
            if (!dropdown || !select) return;
            const value = String(select.value || '');
            const options = dropdown.querySelectorAll('.circuit-option');
            let activeIndex = 0;
            options.forEach((opt, idx) => {
                const v = String(opt.dataset.value || '');
                if (v === value) {
                    opt.classList.add('selected');
                    activeIndex = idx;
                } else {
                    opt.classList.remove('selected');
                }
            });
            this._eventActiveIndex = activeIndex;
        },

        async updateSessionInfo(eventId) {
            const startBtn = this.elements['start-sim-btn'];
            const status = this.elements['connection-status'];
            
            if (!eventId) {
                if (startBtn) startBtn.disabled = true;
                F1Sim.UI.renderGridPreview([]);
                return;
            }

            if (startBtn) startBtn.disabled = true;
            if (status) status.textContent = "Verificando disponibilidad de datos...";

            try {
                // Fetch session info and drivers (grid)
                const [info, drivers] = await Promise.all([
                    F1Sim.API.fetchSessionInfo(eventId),
                    F1Sim.API.fetchDrivers(eventId)
                ]);
                
                if (info && typeof info.error === 'string') {
                    if (status) {
                        status.textContent = "Datos no disponibles para este evento.";
                        status.style.color = "#f57f17";
                    }
                    return;
                }

                // Show grid preview
                F1Sim.UI.renderGridPreview(drivers);

                const laps = Number(info && info.total_laps);
                if (Number.isFinite(laps) && laps > 0) {
                    F1Sim.State.config.officialLaps = Math.trunc(laps);
                    if (status) {
                        status.textContent = "Datos cargados correctamente. La simulación está lista para iniciar";
                        status.style.color = "#4caf50";
                    }
                    if (startBtn) startBtn.disabled = false;
                } else {
                    if (status) {
                        status.textContent = "Datos de vueltas insuficientes.";
                         status.style.color = "#f57f17";
                    }
                }
            } catch (err) {
                if (status) {
                    status.textContent = "Error al verificar datos del evento.";
                    status.style.color = "#ff5252";
                }
                console.error(err);
            }
        },

        setupEventListeners() {
            const el = this.elements;

            if (el['event-select-input']) {
                el['event-select-input'].addEventListener('focus', () => {
                    this.openEventDropdown();
                });
                el['event-select-input'].addEventListener('input', (e) => {
                    this.filterEventDropdown(e.target.value);
                    this.openEventDropdown();
                });
                el['event-select-input'].addEventListener('keydown', (e) => {
                    if (e.key === 'ArrowDown') {
                        e.preventDefault();
                        this.openEventDropdown();
                        this.moveActiveOption(1);
                    } else if (e.key === 'ArrowUp') {
                        e.preventDefault();
                        this.openEventDropdown();
                        this.moveActiveOption(-1);
                    } else if (e.key === 'Enter') {
                        e.preventDefault();
                        this.selectActiveOption();
                    } else if (e.key === 'Escape') {
                        e.preventDefault();
                        this.closeEventDropdown();
                    }
                });
            }

            if (el['event-select-toggle']) {
                el['event-select-toggle'].addEventListener('click', () => {
                    const dropdown = el['event-select-dropdown'];
                    if (!dropdown) return;
                    const isOpen = dropdown.classList.contains('open');
                    if (isOpen) this.closeEventDropdown();
                    else {
                        this.openEventDropdown();
                        if (el['event-select-input']) el['event-select-input'].focus();
                    }
                });
            }

            document.addEventListener('click', (ev) => {
                const combo = el['event-select-combobox'];
                if (!combo) return;
                if (!combo.contains(ev.target)) {
                    this.closeEventDropdown();
                }
            });

            if (el['event-select']) {
                el['event-select'].addEventListener('change', async (e) => {
                    const eventId = e.target.value;

                    if (eventId === 'retry') {
                        this.loadEvents(true);
                        return;
                    }

                    F1Sim.State.config.event = eventId;
                    
                    const opt = e.target.selectedOptions[0];
                    if (opt && opt.dataset) {
                         F1Sim.State.config.eventMeta = {
                            name: opt.dataset.eventName,
                            country: opt.dataset.country,
                            location: opt.dataset.location
                         };
                    }
                    
                    if (eventId) {
                        await this.updateSessionInfo(eventId);
                    } else {
                        if (el['start-sim-btn']) el['start-sim-btn'].disabled = true;
                        if (el['connection-status']) el['connection-status'].textContent = "";
                    }
                });
            }

            if (el['start-sim-btn']) {
                el['start-sim-btn'].addEventListener('click', () => {
                    F1Sim.Invoker.execute(new F1Sim.Commands.StartSimulation(F1Sim.Engine));
                });
            }

            if (el['reimport-btn']) {
                el['reimport-btn'].addEventListener('click', async () => {
                    if (!confirm('¿Desea re-importar los datos de 2026? Esto actualizará la caché y puede tardar unos momentos.')) return;
                    
                    const btn = el['reimport-btn'];
                    const originalText = btn.textContent;
                    btn.disabled = true;
                    btn.textContent = 'Actualizando...';
                    
                    try {
                        const res = await F1Sim.API.reimportData();
                        if (res.status === 'ok' || res.status === 'partial_success') {
                            let msg = 'Datos actualizados correctamente. Se recargarán los eventos.';
                            if (res.status === 'partial_success') {
                                msg = res.message || 'Actualización parcial.';
                                if (res.errors && res.errors.length > 0) {
                                    const maxErrors = 5;
                                    const errorDetails = res.errors.slice(0, maxErrors).map(e => `- ${e.location}: ${e.error}`).join('\n');
                                    msg += `\n\nErrores (${res.errors.length}):\n${errorDetails}`;
                                    if (res.errors.length > maxErrors) msg += `\n... y ${res.errors.length - maxErrors} más.`;
                                }
                            }
                            alert(msg);
                            await this.loadEvents(true);
                            if (el['drivers-list']) el['drivers-list'].innerHTML = '';
                            if (el['event-select']) el['event-select'].value = '';
                        } else {
                            alert('Error al actualizar: ' + (res.message || 'Desconocido') + (res.error ? '\nDetalle: ' + res.error : ''));
                        }
                    } catch (e) {
                        alert('Error de conexión al actualizar datos');
                    } finally {
                        btn.disabled = false;
                        btn.textContent = originalText;
                    }
                });
            }

            if (el['play-pause-btn']) {
                el['play-pause-btn'].addEventListener('click', () => {
                    F1Sim.Invoker.execute(new F1Sim.Commands.TogglePause(F1Sim.Engine));
                });
            }

            if (el['time-scrubber']) {
                const beginScrub = (target, pointerId) => {
                    const s = F1Sim.State.simState;
                    const now = F1Sim.Utils.getNowMs();

                    if (typeof s._scrubWasRunning !== 'boolean') s._scrubWasRunning = false;
                    s._scrubWasRunning = !!s.running;
                    s.scrubbing = true;

                    if (s.clock) {
                        if (s.running) {
                            s.clock.setRunning(false, now);
                            s.running = false;
                        }
                        s.clock.syncNow(now);
                    }

                    F1Sim.UI.updateUI();
                };

                const endScrub = (target, pointerId) => {
                    const s = F1Sim.State.simState;
                    const now = F1Sim.Utils.getNowMs();

                    s.scrubbing = false;

                    const shouldResume = !!s._scrubWasRunning && !s.finished;
                    s._scrubWasRunning = false;

                    if (s.clock) s.clock.syncNow(now);
                    if (shouldResume && s.clock) {
                        s.clock.setRunning(true, now);
                        s.running = true;
                        F1Sim.Engine.loop();
                    }

                    F1Sim.UI.updateUI();
                };

                const scrubber = el['time-scrubber'];
                const supportsPointer = typeof window !== 'undefined' && typeof window.PointerEvent !== 'undefined';
                let active = false;

                const endFromWindow = () => {
                    if (!active) return;
                    active = false;
                    if (supportsPointer) {
                        window.removeEventListener('pointerup', endFromWindow, true);
                        window.removeEventListener('pointercancel', endFromWindow, true);
                    } else {
                        window.removeEventListener('mouseup', endFromWindow, true);
                        window.removeEventListener('touchend', endFromWindow, true);
                        window.removeEventListener('touchcancel', endFromWindow, true);
                    }
                    endScrub(scrubber, NaN);
                };

                if (supportsPointer) {
                    scrubber.addEventListener('pointerdown', (e) => {
                        active = true;
                        beginScrub(e.currentTarget, e.pointerId);
                        window.addEventListener('pointerup', endFromWindow, true);
                        window.addEventListener('pointercancel', endFromWindow, true);
                    });
                } else {
                    scrubber.addEventListener('mousedown', (e) => {
                        active = true;
                        beginScrub(e.currentTarget, NaN);
                        window.addEventListener('mouseup', endFromWindow, true);
                    });
                    scrubber.addEventListener('touchstart', (e) => {
                        active = true;
                        beginScrub(e.currentTarget, NaN);
                        window.addEventListener('touchend', endFromWindow, true);
                        window.addEventListener('touchcancel', endFromWindow, true);
                    }, { passive: true });
                }
                el['time-scrubber'].addEventListener('input', (e) => {
                    const val = Number(e.target.value);
                    if (Number.isFinite(val)) {
                        F1Sim.Invoker.execute(new F1Sim.Commands.SetTime(F1Sim.Engine, val, true));
                    }
                });
                el['time-scrubber'].addEventListener('change', (e) => {
                    const val = Number(e.target.value);
                    if (Number.isFinite(val)) {
                        F1Sim.Invoker.execute(new F1Sim.Commands.SetTime(F1Sim.Engine, val, false));
                    }
                });
            }

            if (el['speed-select']) {
                el['speed-select'].addEventListener('change', (e) => {
                    const spd = parseFloat(e.target.value);
                    if (Number.isFinite(spd) && spd >= 0) {
                        F1Sim.Invoker.execute(new F1Sim.Commands.ChangeSpeed(F1Sim.Engine, spd));
                    }
                });
            }

            if (el['view-mode-select']) {
                el['view-mode-select'].addEventListener('change', (e) => {
                    F1Sim.State.simState.viewMode = e.target.value;
                    F1Sim.Renderer.draw();
                });
            }

            if (el['rotation-slider']) {
                el['rotation-slider'].addEventListener('input', (e) => {
                    F1Sim.State.simState.rotation = parseInt(e.target.value, 10);
                    F1Sim.Renderer.draw();
                });
            }

            if (el['new-sim-btn']) {
                el['new-sim-btn'].addEventListener('click', () => {
                    F1Sim.Invoker.execute(new F1Sim.Commands.StopSimulation(F1Sim.Engine));
                    if (el['config-panel']) el['config-panel'].classList.remove('hidden');
                    if (el['simulation-panel']) el['simulation-panel'].classList.add('hidden');
                });
            }
            
            // Leaderboard controls
            if (el['metrics-modal'] && el['close-metrics-btn']) {
                el['close-metrics-btn'].addEventListener('click', () => {
                    el['metrics-modal'].classList.add('hidden');
                });
            }

            if (el['close-metrics-btn']) {
                el['close-metrics-btn'].addEventListener('click', () => {
                    if (el['metrics-modal']) el['metrics-modal'].style.display = 'none';
                });
            }
        },

        setupCanvasInteractions() {
            const canvas = this.elements['race-canvas'];
            if (!canvas) return;

            const s = F1Sim.State.simState;
            const el = this.elements;

            if (el['zoom-in']) {
                el['zoom-in'].addEventListener('click', () => {
                    s.zoom = Math.min(10, s.zoom * 1.1);
                    F1Sim.Renderer.draw();
                });
            }
            if (el['zoom-out']) {
                el['zoom-out'].addEventListener('click', () => {
                    s.zoom = Math.max(0.1, s.zoom / 1.1);
                    F1Sim.Renderer.draw();
                });
            }
            if (el['zoom-reset']) {
                el['zoom-reset'].addEventListener('click', () => {
                    s.zoom = 1.0;
                    s.panX = 0;
                    s.panY = 0;
                    s.rotation = 0;
                    if (el['rotation-slider']) el['rotation-slider'].value = '0';
                    F1Sim.Renderer.draw();
                });
            }

            canvas.addEventListener('mousedown', (e) => {
                s.isDragging = true;
                s.lastMouseX = e.clientX;
                s.lastMouseY = e.clientY;
                canvas.style.cursor = 'grabbing';
            });

            window.addEventListener('mouseup', () => {
                s.isDragging = false;
                if (canvas) canvas.style.cursor = 'grab';
            });

            canvas.addEventListener('mousemove', (e) => {
                const rect = canvas.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;
                
                // Tooltip logic
                const driver = F1Sim.Renderer.getDriverAtPosition(mouseX, mouseY);
                this.updateTooltip(driver, e.clientX, e.clientY);

                // Pan logic
                if (s.isDragging) {
                    const dx = e.clientX - s.lastMouseX;
                    const dy = e.clientY - s.lastMouseY;
                    s.panX += dx;
                    s.panY += dy;
                    s.lastMouseX = e.clientX;
                    s.lastMouseY = e.clientY;
                    F1Sim.Renderer.draw();
                }
            });

            canvas.addEventListener('wheel', (e) => {
                e.preventDefault();
                const zoomIntensity = 0.1;
                const delta = e.deltaY < 0 ? 1 : -1;
                const newZoom = s.zoom * (1 + delta * zoomIntensity);
                s.zoom = Math.min(Math.max(0.1, newZoom), 10);
                F1Sim.Renderer.draw();
            }, { passive: false });
        },
        
        updateTooltip(driver, x, y) {
            const tooltip = this.elements['tooltip'];
            if (!tooltip) return;
            
            if (driver) {
                tooltip.style.display = 'block';
                tooltip.style.left = (x + 10) + 'px';
                tooltip.style.top = (y + 10) + 'px';
                tooltip.innerHTML = `
                    <strong>${driver.name}${driver.number ? ' #' + driver.number : ''}</strong><br>
                    ${driver.team}<br>
                    Pos: ${driver.pos}<br>
                    Tyre: ${driver.tyre}
                `;
            } else {
                tooltip.style.display = 'none';
            }
        },

        toggleSimulation() {
            const s = F1Sim.State.simState;
            const btn = this.elements['play-pause-btn'];
            
            if (s.finished) {
                // Auto-rewind to start if finished and user clicks "Restart"
                this.setTime(0, false);
                // setTime sets finished=false if time < maxTime, but let's ensure it
                s.finished = false;
            }

            const now = F1Sim.Utils.getNowMs();
            if (s.running) {
                s.clock.setRunning(false, now);
                s.running = false;
                if (btn) btn.textContent = '▶ Reanudar';
            } else {
                s.clock.setRunning(true, now);
                s.running = true;
                if (btn) btn.textContent = '❚❚ Pausa';
                F1Sim.Engine.loop();
            }
        },

        stopSimulation() {
            const s = F1Sim.State.simState;
            if (s.clock) s.clock.setRunning(false);
            s.running = false;
            s.finished = false;
            if (this.elements['play-pause-btn']) this.elements['play-pause-btn'].textContent = '▶ Iniciar';
        },

        setSimulationTime(time, isScrubbing) {
            const s = F1Sim.State.simState;
            s.scrubbing = isScrubbing;
            
            if (s.clock) {
                s.clock.setTime(time, F1Sim.Utils.getNowMs());
                s.time = s.clock.time;
            } else {
                s.time = time;
            }
            
            F1Sim.Engine.updateSimulationLogic(0);
            F1Sim.Renderer.draw();
            this.updateUI();
        },

        updateSpeedDisplay() {
            const s = F1Sim.State.simState;
            if (this.elements['speed-val']) {
                const sp = Number.isFinite(s.speedTarget) ? s.speedTarget : 0;
                this.elements['speed-val'].textContent = sp.toFixed(sp % 1 === 0 ? 0 : 1) + 'x';
            }
            if (this.elements['speed-select']) {
                const sp = Number.isFinite(s.speedTarget) ? s.speedTarget : 0;
                const v = sp.toFixed(1);
                if (this.elements['speed-select'].value !== v) this.elements['speed-select'].value = v;
            }
            const status = this.elements['sim-status-indicator'];
            if (status) {
                if (s.finished) {
                    status.textContent = 'FIN';
                    status.classList.remove('active');
                    status.classList.remove('paused');
                } else if (s.running) {
                    status.textContent = 'ON';
                    status.classList.add('active');
                    status.classList.remove('paused');
                } else {
                    status.textContent = (Number.isFinite(s.speedTarget) && s.speedTarget === 0) ? 'PAUSA' : 'OFF';
                    status.classList.remove('active');
                    status.classList.toggle('paused', status.textContent === 'PAUSA');
                }
            }
        },

        updateControls() {
            const s = F1Sim.State.simState;
            const btn = this.elements['play-pause-btn'];
            if (btn) {
                if (s.finished) {
                    btn.textContent = '↺ Reiniciar';
                } else if (s.running) {
                    btn.textContent = '❚❚ Pausa';
                } else {
                    btn.textContent = '▶ Reanudar';
                }
            }
        },

        updateUI() {
            this.updateSpeedDisplay();
            this.updateControls();
            const s = F1Sim.State.simState;
            
            // Time displays
            if (this.elements['timer']) this.elements['timer'].textContent = F1Sim.Utils.formatTime(s.time);
            
            if (this.elements['time-scrubber']) {
                this.elements['time-scrubber'].max = s.maxTime;
                if (!s.scrubbing) this.elements['time-scrubber'].value = s.time;
                if (Number.isFinite(s.maxTime) && s.maxTime > 0) {
                    const pct = Math.max(0, Math.min(100, (s.time / s.maxTime) * 100));
                    this.elements['time-scrubber'].style.setProperty('--progress', `${pct}%`);
                } else {
                    this.elements['time-scrubber'].style.setProperty('--progress', `0%`);
                }
            }

            if (this.elements['timer-remaining']) {
                const remaining = Math.max(0, s.maxTime - s.time);
                this.elements['timer-remaining'].textContent = '-' + F1Sim.Utils.formatTime(remaining);
            }

            // Lap counters
            const maxLaps = F1Sim.State.config.officialLaps || (F1Sim.State.simData && F1Sim.State.simData.total_laps) || 60;
            let currentLap = 1;
            if (Array.isArray(s.positions) && s.positions.length > 0) {
                const leader = s.positions[0];
                const leaderLap = Number(leader.lapsCompleted) + 1;
                if (Number.isFinite(leaderLap) && leaderLap > 0) currentLap = leaderLap;
            } else if (s.maxTime > 0) {
                const progress = s.time / s.maxTime;
                currentLap = Math.ceil(progress * maxLaps) || 1;
            }
            currentLap = Math.min(maxLaps, Math.max(1, currentLap));
            
            if (this.elements['current-lap']) this.elements['current-lap'].textContent = currentLap;
            if (this.elements['total-laps']) this.elements['total-laps'].textContent = maxLaps;

            this.updateLeaderboard();
            this.updateRaceSummary();
        },

        updateLeaderboard() {
            const container = document.querySelector('#leaderboard .leaderboard-container');
            if (!container) return;
            const s = F1Sim.State.simState;

            // Get current driver states sorted by position
            let drivers = [];
            if (F1Sim.State.simData && F1Sim.State.simData.drivers) {
                 drivers = Object.values(F1Sim.State.simData.drivers).map(d => {
                     // Find dynamic state
                     const dyn = F1Sim.State.simState.positions.find(p => p.driver_id === d.driver_id) || {};
                     return { ...d, ...dyn };
                 });
            }
            
            // Calculate KPI & Position Change FIRST (before sorting)
            if (drivers.length > 0) {
                // Determine current race positions for gap calculation
                // Note: The array might be unsorted initially, but 'pos' field should be reliable if coming from simState
                // If not, we should sort by 'pos' first to establish current standing
                drivers.sort((a, b) => (a.pos || 99) - (b.pos || 99));

                const leader = drivers[0];
                const leaderProg = (leader.lapsCompleted || 0) + (leader.currentProgress || 0);
                
                const refs = (s && s.positionChangeRef) || {};

                drivers.forEach((d, idx) => {
                    const currentPos = d.pos || (idx + 1);
                    const ref =
                        refs[String(d.driver_id)] ||
                        refs[String(d.DriverId)] ||
                        null;

                    let basePos;
                    if (ref && Number.isFinite(ref.pos)) {
                        basePos = ref.pos;
                    } else {
                        const gridPos = d.GridPosition || currentPos;
                        basePos = gridPos;
                    }

                    const posChange = basePos - currentPos; 
                    d.posChange = posChange;
                    
                    // --- Gap & Interval Logic ---
                    // Helper to format time
                    const fmt = (t) => {
                        if (t >= 60) {
                            const m = Math.floor(t / 60);
                            const s = (t % 60).toFixed(1);
                            return `${m}:${s.padStart(4, '0')}`; 
                        }
                        return t.toFixed(1);
                    };

                    let refLapTime = 90; 
                    if (d.laps && d.laps.length > 0) {
                        const lapIdx = Math.max(0, Math.min(d.laps.length - 1, (d.lapsCompleted || 1) - 1));
                        if (d.laps[lapIdx] && d.laps[lapIdx].time) refLapTime = d.laps[lapIdx].time;
                    }

                    if (idx === 0) {
                        d.gapStr = 'Leader';
                        d.intStr = '-';
                    } else if (d.finished && leader.finished) {
                        const diffL = d.totalTime - leader.totalTime;
                        d.gapStr = `+${fmt(diffL)}`;
                        const ahead = drivers[idx-1];
                        if (ahead.finished) {
                            const diffA = d.totalTime - ahead.totalTime;
                            d.intStr = `+${fmt(diffA)}`;
                        } else d.intStr = '-';
                    } else {
                        const dProg = (d.lapsCompleted || 0) + (d.currentProgress || 0);
                        const diffL = Math.max(0, leaderProg - dProg);
                        const gapTime = diffL * refLapTime;
                        d.gapStr = gapTime > 0 ? `+${fmt(gapTime)}` : '-';
                        
                        const ahead = drivers[idx-1];
                        const aheadProg = (ahead.lapsCompleted || 0) + (ahead.currentProgress || 0);
                        const diffA = Math.max(0, aheadProg - dProg);
                        const intTime = diffA * refLapTime;
                        d.intStr = intTime > 0 ? `+${fmt(intTime)}` : '-';
                    }
                });
            }

            // Apply Sorting based on user selection
            const sortKey = F1Sim.State.simState.leaderboardSort || 'pos';
            const sortDir = F1Sim.State.simState.leaderboardSortDir || 1;

            drivers.sort((a, b) => {
                let valA, valB;
                switch(sortKey) {
                    case 'name': 
                        valA = (a.FullName || a.Abbreviation || '').toLowerCase(); 
                        valB = (b.FullName || b.Abbreviation || '').toLowerCase();
                        break;
                    case 'team':
                        valA = (a.TeamName || a.team || '').toLowerCase();
                        valB = (b.TeamName || b.team || '').toLowerCase();
                        break;
                    case 'change':
                        // Sort by gains (highest positive first)
                        valA = a.posChange || 0;
                        valB = b.posChange || 0;
                        break;
                    default: // 'pos'
                        valA = a.pos || 99;
                        valB = b.pos || 99;
                }
                
                if (valA < valB) return -1 * sortDir;
                if (valA > valB) return 1 * sortDir;
                return 0;
            });
            
            container.innerHTML = '';
            drivers.forEach(d => {
                const row = document.createElement('div');
                row.className = 'leaderboard-row';

                // We keep the row coloring for big movers if desired, 
                // but user asked for specific cell formatting. 
                // We can keep the row highlight as an extra visual cue.
                if (typeof d.posChange === 'number') {
                    if (d.posChange > 0) row.classList.add('pos-up');
                    else if (d.posChange < 0) row.classList.add('pos-down');
                }
                
                const dorsal = d.Number || d.number || d.carNumber || '';
                const teamColor = d.color || '#333';
                const dorsalPick = F1Sim.Utils.pickReadableTextColor(teamColor);
                const dorsalTextColor = dorsalPick.color;
                const dorsalStyle = [
                    `background:${teamColor}`,
                    `color:${dorsalTextColor}`,
                    'border-radius:999px',
                    'border:1px solid rgba(0,0,0,0.6)',
                    'display:inline-flex',
                    'align-items:center',
                    'justify-content:center',
                    'min-width:24px',
                    'padding:0 6px',
                    'font-weight:700'
                ].join(';');
                const posClass = d.pos === 1 ? 'p1' : d.pos === 2 ? 'p2' : d.pos === 3 ? 'p3' : '';
                
                let changeClass = 'neutral';
                let changeStr = '-';
                let changeIcon = '-'; // Default neutral icon

                if (typeof d.posChange === 'number') {
                    if (d.posChange > 0) {
                        changeClass = 'gain';
                        changeStr = `+${d.posChange}`;
                        changeIcon = '▲';
                    } else if (d.posChange < 0) {
                        changeClass = 'loss';
                        changeStr = `${d.posChange}`; // Negative sign included
                        changeIcon = '▼';
                    } else {
                        changeClass = 'neutral';
                        changeStr = '-';
                        changeIcon = '=';
                    }
                }
                
                // Tooltip for change
                const changeTitle = `Salida: P${d.GridPosition || '?'} | Actual: P${d.pos}`;

                row.innerHTML = `
                    <div class="lb-pos ${posClass}">${d.pos || '-'}</div>
                    <div class="lb-name" title="${d.FullName || d.Abbreviation}">
                         ${d.FullName || d.Abbreviation}
                    </div>
                    <div class="lb-change ${changeClass}" title="${changeTitle}">
                        <span class="change-icon">${changeIcon}</span>
                        <span class="change-value">${changeStr}</span>
                    </div>
                    <div class="lb-team">${d.team || d.TeamName || ''}</div>
                    <div class="lb-dorsal" style="${dorsalStyle}">${dorsal}</div>
                `;
                row.addEventListener('click', () => this.showDriverMetrics(d));
                container.appendChild(row);
            });
        },

        updateRaceSummary() {
            const s = F1Sim.State.simState;
            const data = F1Sim.State.simData;
            const summaryEl = this.elements['race-summary'];
            const bodyEl = document.getElementById('race-summary-body');
            // Chart canvas removed from logic as it's removed from UI
            if (!summaryEl || !bodyEl) return;
            if (!data || !data.drivers) {
                summaryEl.classList.add('hidden');
                return;
            }

            // Only show if race is finished
            if (!s.finished) {
                summaryEl.classList.add('hidden');
                return;
            }

            const snapshot = Array.isArray(s.positions) && s.positions.length
                ? [...s.positions].sort((a, b) => (a.pos || 99) - (b.pos || 99))
                : null;
            if (!snapshot || !snapshot.length) {
                summaryEl.classList.add('hidden');
                return;
            }

            const driversMap = data.drivers;
            
            // Build driver lookup map by driver_id for safe access
            const driversById = {};
            if (driversMap) {
                Object.values(driversMap).forEach(d => {
                    driversById[d.driver_id] = d;
                });
            }

            const rows = snapshot.map((p, idx) => {
                const base = driversById[p.driver_id];
                if (!base) return null;
                const startPos = Number.isFinite(base.GridPosition) && base.GridPosition > 0 ? base.GridPosition : (idx + 1);
                const finishPos = p.pos;
                const delta = startPos - finishPos; // Positive = Gain, Negative = Loss
                return {
                    driverId: p.driver_id,
                    pos: finishPos,
                    startPos,
                    finishPos,
                    delta,
                    name: base.FullName || base.Abbreviation || p.driver_id,
                    color: base.color || '#888'
                };
            }).filter(Boolean).sort((a, b) => a.pos - b.pos);

            if (!rows.length) {
                summaryEl.classList.add('hidden');
                return;
            }

            // Validation Logic
            let deltaSum = 0;
            const finishPositions = new Set();
            rows.forEach(r => {
                deltaSum += r.delta;
                finishPositions.add(r.finishPos);
            });
            
            // Check if sum is zero (algebraic sum of differences)
            // Note: This holds true if startPos and finishPos are permutations of the same set {1..N}
            if (deltaSum !== 0) {
                console.warn(`[RaceSummary] Validation Warning: Sum of deltas is ${deltaSum} (Expected 0)`);
            }
            if (finishPositions.size !== rows.length) {
                console.warn(`[RaceSummary] Validation Warning: Duplicate finish positions detected.`);
            }

            summaryEl.classList.remove('hidden');
            bodyEl.innerHTML = '';

            rows.forEach(r => {
                const tr = document.createElement('tr');
                
                let deltaClass = 'pos-neutral';
                let deltaIcon = '';
                let deltaText = '0';
                
                if (r.delta > 0) {
                    deltaClass = 'pos-gain';
                    deltaIcon = '▲';
                    deltaText = `+${r.delta}`;
                } else if (r.delta < 0) {
                    deltaClass = 'pos-loss';
                    deltaIcon = '▼';
                    deltaText = `${r.delta}`;
                }

                // Columns: Pilot, Start, Finish, Delta
                tr.innerHTML = `
                    <td>
                        <div style="display:flex;align-items:center;gap:8px">
                            <span style="width:4px;height:16px;background:${r.color};display:inline-block;border-radius:2px"></span>
                            ${r.name}
                        </div>
                    </td>
                    <td>${r.startPos}</td>
                    <td>${r.finishPos}</td>
                    <td class="${deltaClass}">${deltaIcon} ${deltaText}</td>
                `;
                bodyEl.appendChild(tr);
            });
        },

        showDriverMetrics(driver) {
            const modal = this.elements['metrics-modal'];
            const content = this.elements['metrics-content'];
            if (!modal || !content) return;
            
            // Try to find validation data
            let valData = null;
            if (F1Sim.State.simData && F1Sim.State.simData.metrics && 
                F1Sim.State.simData.metrics.validation && 
                F1Sim.State.simData.metrics.validation.drivers_vs_history) {
                valData = F1Sim.State.simData.metrics.validation.drivers_vs_history[driver.Abbreviation];
            }

            let metricsHtml = '';
            
            const source = driver.stats_source || 'Unknown';
            let sourceColor = '#aaa';
            if (source.includes('High')) sourceColor = '#4cd964';
            else if (source.includes('Medium')) sourceColor = '#ffcc00';
            else if (source.includes('Low')) sourceColor = '#ff3b30';

            const sourceHtml = `
                <div class="metrics-source-card">
                    <span class="metrics-source-label metric-tooltip" title="Basado en históricos por piloto, equipo o media global según disponibilidad.">
                        Calidad de Datos
                    </span>
                    <span class="metrics-source-value" style="color:${sourceColor};">
                        ${source}
                    </span>
                </div>
            `;

            if (valData) {
                const deltaVal = valData.delta;
                const isFaster = deltaVal < 0;
                const color = isFaster ? '#4cd964' : '#ff3b30';
                const sign = deltaVal > 0 ? '+' : '';

                metricsHtml = `
                    <div class="metrics-accordion">
                        <button class="metrics-accordion-header" type="button" data-section="validation" aria-expanded="true">
                            Validación de Predicción (vs Histórico)
                            <span class="metrics-accordion-icon">▾</span>
                        </button>
                        <div class="metrics-accordion-panel open" data-section="validation">
                            <div class="metrics-validation-grid">
                                <div class="metrics-validation-item">
                                    <div class="metrics-card-label">Predicho (Avg)</div>
                                    <div class="metrics-card-value-md metric-tooltip" title="Promedio de vueltas simuladas para este piloto.">
                                        ${valData.predicted_avg_lap}s
                                    </div>
                                </div>
                                <div class="metrics-validation-item">
                                    <div class="metrics-card-label">Histórico (Avg)</div>
                                    <div class="metrics-card-value-md metric-tooltip" title="Promedio de vueltas reales históricas en este circuito.">
                                        ${valData.historical_avg_lap}s
                                    </div>
                                </div>
                                <div class="metrics-validation-item">
                                    <div class="metrics-card-label">Delta</div>
                                    <div class="metrics-card-value-md metric-tooltip" style="color:${color};" title="Diferencia entre ritmo simulado y histórico. Negativo = más rápido.">
                                        ${sign}${deltaVal}s
                                    </div>
                                </div>
                            </div>
                            <div class="metrics-validation-note">
                                Delta negativo indica ritmo predicho más rápido que el histórico.
                            </div>
                        </div>
                    </div>
                `;
            } else {
                metricsHtml = `
                    <div class="metrics-accordion">
                        <button class="metrics-accordion-header" type="button" data-section="validation" aria-expanded="true">
                            Validación de Predicción (vs Histórico)
                            <span class="metrics-accordion-icon">▾</span>
                        </button>
                        <div class="metrics-accordion-panel open" data-section="validation">
                            <p class="metrics-validation-note">
                                Sin datos históricos suficientes para validar esta predicción.
                            </p>
                        </div>
                    </div>
                `;
            }
            
            content.innerHTML = `
                <div class="metrics-layout">
                    <div class="metrics-header">
                        <h3 class="metrics-header-title">
                            ${driver.FullName}
                            <span class="metrics-header-sub">
                                #${driver.BroadcastName || driver.carNumber || driver.Abbreviation || ''}
                            </span>
                        </h3>
                        <div class="metrics-header-team" style="color:${driver.color || '#fff'};">
                            ${driver.TeamName || ''}
                        </div>
                    </div>
                    
                    ${sourceHtml}

                    <div class="metrics-summary-grid">
                       <div>
                           <div class="metrics-card-label">Posición Actual</div>
                           <div class="metrics-card-value-lg">P${driver.pos || '-'}</div>
                       </div>
                       <div>
                           <div class="metrics-card-label">Vueltas</div>
                           <div class="metrics-card-value-lg">${driver.lapsCompleted || 0}</div>
                       </div>
                       <div>
                           <div class="metrics-card-label">Mejor Vuelta</div>
                           <div class="metrics-card-value-md">
                               ${driver.bestLap || '-'}
                           </div>
                       </div>
                        <div>
                           <div class="metrics-card-label">Gap Leader</div>
                           <div class="metrics-card-value-mono">
                               ${driver.gapStr || '-'}
                           </div>
                       </div>
                    </div>

                    ${metricsHtml}
                </div>
            `;
            modal.style.display = 'block';
            
            const headers = content.querySelectorAll('.metrics-accordion-header');
            headers.forEach(btn => {
                btn.addEventListener('click', () => {
                    const section = btn.getAttribute('data-section');
                    const panel = content.querySelector('.metrics-accordion-panel[data-section="' + section + '"]');
                    if (!panel) return;
                    const isOpen = panel.classList.toggle('open');
                    btn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
                });
            });
        },
    },

    // --- Renderer Module ---
    Renderer: {
        draw() {
            const ctx = F1Sim.UI.ctx;
            const canvas = F1Sim.UI.elements['race-canvas'];
            const s = F1Sim.State.simState;
            
            if (!ctx || !canvas) return;

            // Clear
            ctx.fillStyle = '#1e1e1e';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.save();
            
            // Apply Transformations
            ctx.translate(canvas.width / 2, canvas.height / 2);
            ctx.translate(s.panX, s.panY);
            ctx.scale(s.zoom, s.zoom);
            if (s.viewMode === 'abstract') {
                // Abstract view logic
            } else {
                ctx.rotate(s.rotation * Math.PI / 180);
            }

            this.drawTrack(ctx);
            this.drawCars(ctx);

            ctx.restore();
        },

        drawTrack(ctx) {
            const pathData = F1Sim.State.simData && F1Sim.State.simData.circuit_path;
            const layout = F1Sim.State.simState.trackLayout;
            
            if (!pathData || !layout) return;
            const mode = F1Sim.State.simState.colorMode === 'daltonismo' ? 'daltonismo' : 'default';
            const vis = F1Sim.Constants.VISUAL_CONFIG;
            const pal = (vis && vis.paletas && vis.paletas[mode]) ? vis.paletas[mode] : null;
            const sectorColors = pal && Array.isArray(pal.sectores) && pal.sectores.length === 3 ? pal.sectores : F1Sim.Constants.SECTOR_COLORS;
            const baseColor = (pal && pal.pista_base) ? pal.pista_base : '#2a2a2a';
            const centerColor = (pal && pal.linea_central) ? pal.linea_central : '#d0d0d0';
            
            ctx.save();
            // Center the track and apply base scale
            ctx.scale(layout.baseScale, layout.baseScale);
            ctx.translate(-layout.centerX, -layout.centerY);
            
            const w = (F1Sim.Constants.VISUAL_CONFIG && F1Sim.Constants.VISUAL_CONFIG.coche && Number.isFinite(F1Sim.Constants.VISUAL_CONFIG.coche.radio_px))
                ? F1Sim.Constants.VISUAL_CONFIG.coche.radio_px
                : 18;
            const trackWidthPx = 44 + (w - 18) * 0.6;
            const baseWidth = trackWidthPx / layout.baseScale;
            const sectorWidth = 40 / layout.baseScale;
            const centerWidth = 2 / layout.baseScale;

            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';

            ctx.beginPath();
            ctx.strokeStyle = baseColor;
            ctx.lineWidth = baseWidth;
            if (pathData.length > 0) {
                ctx.moveTo(pathData[0].x, pathData[0].y);
                for (let i = 1; i < pathData.length; i++) ctx.lineTo(pathData[i].x, pathData[i].y);
            }
            ctx.stroke();

            const len = pathData.length;
            const s1End = Math.max(1, Math.floor(len / 3));
            const s2End = Math.max(s1End + 1, Math.floor((2 * len) / 3));
            const s3End = len - 1;

            const drawSegment = (start, end, color) => {
                ctx.beginPath();
                ctx.strokeStyle = color;
                ctx.lineWidth = sectorWidth;
                ctx.moveTo(pathData[start].x, pathData[start].y);
                for (let i = start + 1; i <= end; i++) ctx.lineTo(pathData[i].x, pathData[i].y);
                ctx.stroke();
            };

            if (len > 2) {
                drawSegment(0, s1End, sectorColors[0]);
                drawSegment(s1End, s2End, sectorColors[1]);
                drawSegment(s2End, s3End, sectorColors[2]);
            }
            
            // Center line
            ctx.beginPath();
            ctx.strokeStyle = centerColor;
            ctx.lineWidth = centerWidth;
            if (pathData.length > 0) {
                ctx.moveTo(pathData[0].x, pathData[0].y);
                for (let i = 1; i < pathData.length; i++) ctx.lineTo(pathData[i].x, pathData[i].y);
            }
            ctx.stroke();
            
            ctx.restore();
        },

        drawCars(ctx) {
            const positions = F1Sim.State.simState.positions;
            const layout = F1Sim.State.simState.trackLayout;
            if (!positions || !layout) return;
            const driverById = F1Sim.State.simState.driverById || {};
            const vis = F1Sim.Constants.VISUAL_CONFIG;
            const zoom = Number.isFinite(F1Sim.State.simState.zoom) ? F1Sim.State.simState.zoom : 1.0;
            const baseScale = Number.isFinite(layout.baseScale) && layout.baseScale > 0 ? layout.baseScale : 1;
            const effectiveScale = baseScale * (zoom > 0 ? zoom : 1.0);
            const carRadiusPx = (vis && vis.coche && Number.isFinite(vis.coche.radio_px)) ? vis.coche.radio_px : 18;
            const borderPx = (vis && vis.coche && Number.isFinite(vis.coche.borde_ancho_px)) ? vis.coche.borde_ancho_px : 2;
            const borderColor = (vis && vis.coche && typeof vis.coche.borde === 'string') ? vis.coche.borde : '#0b0b0b';
            const minDorsalPx = (vis && vis.dorsal && Number.isFinite(vis.dorsal.tam_min_px)) ? vis.dorsal.tam_min_px : 24;
            const dorsalFont = (vis && vis.dorsal && typeof vis.dorsal.tipo_letra === 'string') ? vis.dorsal.tipo_letra : "Arial, sans-serif";
            const overlapBuckets = new Map();

            ctx.save();
            ctx.scale(layout.baseScale, layout.baseScale);
            ctx.translate(-layout.centerX, -layout.centerY);

            positions.forEach(pos => {
                const driver = driverById[String(pos.driver_id)];
                
                if (!driver) return;

                const teamColor = driver.color || '#999999';
                const radius = carRadiusPx / effectiveScale;
                const borderW = borderPx / effectiveScale;
                const fontSize = minDorsalPx / effectiveScale;
                const textPick = F1Sim.Utils.pickReadableTextColor(teamColor);
                let textColor = textPick.color;
                let textBg = textColor === '#ffffff' ? '#000000' : '#ffffff';
                const number = driver.number ?? driver.Number ?? '';

                const key = `${Math.round(pos.x * 10)}:${Math.round(pos.y * 10)}`;
                const slot = overlapBuckets.get(key) || 0;
                overlapBuckets.set(key, slot + 1);

                const angle = (slot % 10) * (Math.PI * 2 / 10);
                const offset = radius * 0.9 * Math.floor(slot / 10 + 1);
                const x = pos.x + Math.cos(angle) * offset;
                const y = pos.y + Math.sin(angle) * offset;

                ctx.fillStyle = teamColor;
                ctx.beginPath();
                ctx.arc(x, y, radius, 0, Math.PI * 2);
                ctx.fill();

                ctx.strokeStyle = borderColor;
                ctx.lineWidth = borderW;
                ctx.stroke();

                ctx.font = `bold ${fontSize}px ${dorsalFont}`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';

                const label = String(number || '').trim() || String(driver.Abbreviation || '').trim();
                if (!label) return;

                if (textPick.contrast < 4.5) {
                    ctx.fillStyle = textBg;
                    ctx.beginPath();
                    ctx.arc(x, y, radius * 0.72, 0, Math.PI * 2);
                    ctx.fill();
                }

                ctx.fillStyle = textColor;
                ctx.fillText(label, x, y);
            });
            
            ctx.restore();
        },
        
        getDriverAtPosition(x, y) {
            const canvas = F1Sim.UI.elements['race-canvas'];
            const s = F1Sim.State.simState;
            const layout = s.trackLayout;
            if (!canvas || !layout || !Array.isArray(s.positions) || s.positions.length === 0) return null;

            let px = x - canvas.width / 2 - (s.panX || 0);
            let py = y - canvas.height / 2 - (s.panY || 0);

            const zoomFactor = Number.isFinite(s.zoom) ? s.zoom : 1.0;
            if (zoomFactor > 0) {
                px /= zoomFactor;
                py /= zoomFactor;
            }

            const rot = (Number.isFinite(s.rotation) ? s.rotation : 0) * Math.PI / 180;
            if (rot !== 0) {
                const c = Math.cos(-rot);
                const si = Math.sin(-rot);
                const rx = px * c - py * si;
                const ry = px * si + py * c;
                px = rx;
                py = ry;
            }

            const baseScale = Number.isFinite(layout.baseScale) && layout.baseScale > 0 ? layout.baseScale : 1;
            px = px / baseScale + layout.centerX;
            py = py / baseScale + layout.centerY;

            const vis = F1Sim.Constants.VISUAL_CONFIG;
            const carRadiusPx = (vis && vis.coche && Number.isFinite(vis.coche.radio_px)) ? vis.coche.radio_px : 18;
            const effectiveScale = baseScale * (zoomFactor > 0 ? zoomFactor : 1.0);
            const r = carRadiusPx / effectiveScale;
            const r2 = r * r;
            let best = null;
            let bestD2 = Infinity;
            for (const p of s.positions) {
                const dx = p.x - px;
                const dy = p.y - py;
                const d2 = dx * dx + dy * dy;
                if (d2 <= r2 && d2 < bestD2) {
                    best = p;
                    bestD2 = d2;
                }
            }
            if (!best) return null;

            const driver = (s.driverById || {})[String(best.driver_id)];
            if (!driver) return null;

            const number = driver.number ?? driver.Number ?? '';
            return {
                name: driver.FullName || driver.Abbreviation || '',
                team: driver.TeamName || driver.team || '',
                pos: best.pos,
                tyre: best.tyre,
                number: String(number || '').trim()
            };
        }
    },

    // --- Engine Module ---
    Engine: {
        setSpeed(speed) {
            const s = F1Sim.State.simState;
            const sp = Number(speed);
            if (!Number.isFinite(sp) || sp < 0) return;

            const now = F1Sim.Utils.getNowMs();
            if (s.clock) {
                if (s.running) {
                    const pre = s.clock.tick(now);
                    s.time = pre.time;
                } else {
                    s.clock.syncNow(now);
                }
            }

            if (sp > 0) s._speedBeforePause = sp;
            s.speedTarget = sp;
            s.forwardSpeed = sp;
            s.reverseSpeed = sp;

            if (s.clock && sp > 0) {
                s.clock.setSpeeds(sp, sp);
            }

            if (sp === 0) {
                if (s.clock) s.clock.setRunning(false, now);
                s.running = false;
            } else if (!s.finished) {
                if (s.clock) {
                    s.clock.setRunning(true, now);
                    if (!s.running) {
                        s.running = true;
                        this.loop();
                    }
                }
            }

            F1Sim.UI.updateSpeedDisplay();
            F1Sim.UI.updateUI();
        },

        togglePause() {
            const s = F1Sim.State.simState;
            if (s.finished) return;
            const now = F1Sim.Utils.getNowMs();
            if (s.running) {
                s.clock.setRunning(false, now);
                s.running = false;
            } else {
                if (!(Number.isFinite(s.speedTarget) && s.speedTarget > 0)) {
                    const restore = Number.isFinite(s._speedBeforePause) && s._speedBeforePause > 0 ? s._speedBeforePause : 1.0;
                    this.setSpeed(restore);
                    return;
                }
                s.clock.setRunning(true, now);
                s.running = true;
                s._lastLoopNowMs = null;
                s._lastSimTimeWhileRunning = s.time;
                this.loop();
            }
            F1Sim.UI.updateUI();
        },

        stop() {
            const s = F1Sim.State.simState;
            if (s.clock) s.clock.setRunning(false);
            s.running = false;
            s.finished = false;
            if (F1Sim.UI.elements['play-pause-btn']) F1Sim.UI.elements['play-pause-btn'].textContent = '▶ Iniciar';
            F1Sim.UI.updateUI();
        },

        setTime(time, isScrubbing) {
            const s = F1Sim.State.simState;
            s.scrubbing = isScrubbing;

            const now = F1Sim.Utils.getNowMs();
            if (s.clock) {
                s.clock.setTime(time, now);
                s.time = s.clock.time;
            } else {
                s.time = time;
            }
            s._lastLoopNowMs = null;
            s._lastSimTimeWhileRunning = s.time;

            if (s.finished && s.time < (s.maxTime - 0.5)) {
                s.finished = false;
                F1Sim.UI.updateControls(); 
            } else if (!s.finished && s.time >= (s.maxTime - 0.1)) {
                s.finished = true;
                F1Sim.UI.updateControls();
            }
            
            this.updateSimulationLogic(0);
            F1Sim.Renderer.draw();
            F1Sim.UI.updateUI();
            F1Sim.Utils.validateState();
        },

        async startSimulation() {
            const el = F1Sim.UI.elements;
            const config = F1Sim.State.config;
            
            if (!config.event) {
                alert('Selecciona un evento');
                return;
            }

            el['start-sim-btn'].disabled = true;
            el['start-sim-btn'].textContent = 'Iniciando...';
            F1Sim.UI.setVisualError(null);

            try {
                let paceScale = 1.0;

                const payload = {
                    event_id: config.event,
                    drivers: (config.drivers || []).map(d => d.driver_id ?? d.DriverId).filter(Boolean),
                    year: config.season,
                    pace_scale: paceScale
                };

                const data = await F1Sim.API.runSimulation(payload);

                if (data.error) {
                    alert('Error: ' + data.error);
                    F1Sim.UI.setVisualError(String(data.error || 'Error al iniciar simulación'));
                    el['start-sim-btn'].disabled = false;
                    el['start-sim-btn'].textContent = 'Iniciar Simulación';
                    return;
                }

                el['config-panel'].classList.add('hidden');
                el['simulation-panel'].classList.remove('hidden');

                // Set text info
                if (el['race-title']) el['race-title'].textContent = `${config.eventMeta.name} ${config.season}`;
                if (el['race-subtitle']) el['race-subtitle'].textContent = config.eventMeta.location;

                requestAnimationFrame(() => {
                    this.initializeSimulation(data);
                });

            } catch (err) {
                alert('Error al iniciar simulación');
                console.error(err);
                F1Sim.UI.setVisualError(err && err.message ? String(err.message) : 'Error al iniciar simulación');
                el['start-sim-btn'].disabled = false;
                el['start-sim-btn'].textContent = 'Iniciar Simulación';
            }
        },

        initializeSimulation(data) {
            const s = F1Sim.State.simState;
            F1Sim.State.simData = data;
            
            // Process Track Data: Center and Normalize
            if (data.circuit_path && data.circuit_path.length > 0) {
                let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
                data.circuit_path.forEach(p => {
                    // Handle case sensitivity and standardize
                    const x = p.X !== undefined ? p.X : p.x;
                    const y = p.Y !== undefined ? p.Y : p.y;
                    p.x = x; 
                    p.y = y;
                    
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                });

                if (data.circuit_path.length > 2) {
                    const first = data.circuit_path[0];
                    const last = data.circuit_path[data.circuit_path.length - 1];
                    const dx = first.x - last.x;
                    const dy = first.y - last.y;
                    const d2 = dx * dx + dy * dy;
                    if (Number.isFinite(d2) && d2 > 1e-6) {
                        data.circuit_path.push({ ...first });
                    }
                }
                
                const width = maxX - minX;
                const height = maxY - minY;
                const maxDim = Math.max(width, height);
                
                // Store layout info for rendering
                s.trackLayout = {
                    minX, maxX, minY, maxY,
                    centerX: (minX + maxX) / 2,
                    centerY: (minY + maxY) / 2,
                    width, height,
                    baseScale: maxDim > 0 ? (800 / maxDim) : 1
                };
            }

            s.driverById = {};
            if (data && data.drivers) {
                Object.values(data.drivers).forEach(d => {
                    if (!d) return;
                    const k = d.driver_id ?? d.DriverId;
                    if (k != null) s.driverById[String(k)] = d;
                });
            }
            
            s.minTime = 0;
            s.maxTime = 0;
            s.time = 0;
            s.finished = false;
            s.finishedDrivers = [];
            s.finalOrderLogged = false;
            s._resumeAfterVisibility = false;
            s._lastLoopNowMs = null;
            s._lastSimTimeWhileRunning = null;
            s.positionChangeRef = {};

            if (data && data.drivers) {
                const lastTimes = Object.values(data.drivers)
                    .map(d => Array.isArray(d.laps) && d.laps.length ? d.laps[d.laps.length - 1].cumulative : 0)
                    .filter(v => Number.isFinite(v) && v > 0);
                if (lastTimes.length) {
                    s.maxTime = Math.max(...lastTimes);
                }
            }

            if (!(Number.isFinite(s.maxTime) && s.maxTime > 0)) {
                const fallbackLaps = Number(data && data.total_laps) || 0;
                s.maxTime = fallbackLaps > 0 ? (fallbackLaps * 90) : 5400;
            }
            
            // Init Clock
            s.clock = new PrecisePlaybackClock({
                minTime: s.minTime,
                maxTime: s.maxTime,
                timeScale: s.timeScale,
                forwardSpeed: s.speedTarget,
                reverseSpeed: s.speedTarget
            });
            
            F1Sim.UI.resizeCanvas();
            const canvas = F1Sim.UI.elements['race-canvas'];
            if (canvas && (!(canvas.width > 0) || !(canvas.height > 0))) {
                F1Sim.UI.setVisualError('Canvas sin tamaño visible; revisa CSS/layout.');
            }
            
            // Start Loop
            const now = F1Sim.Utils.getNowMs();
            const startRunning = Number.isFinite(s.speedTarget) && s.speedTarget > 0;
            s.running = startRunning;
            s.clock.setRunning(startRunning, now);
            F1Sim.UI.updateSpeedDisplay();
            if (startRunning) this.loop();
        },

        loop() {
            const s = F1Sim.State.simState;
            if (!s.running && !s.scrubbing) return;

            requestAnimationFrame(() => this.loop());

            const now = F1Sim.Utils.getNowMs();
            if (Number.isFinite(s._lastLoopNowMs) && s._lastLoopNowMs != null) {
                const dtMs = now - s._lastLoopNowMs;
                if (Number.isFinite(dtMs) && dtMs > 250 && s.clock) {
                    s.clock.syncNow(now);
                }
            }
            s._lastLoopNowMs = now;
            const tick = s.clock.tick(now);
            
            if (s.running) {
                const nextTime = tick.time;
                if (Number.isFinite(s._lastSimTimeWhileRunning) && nextTime < s._lastSimTimeWhileRunning - 0.001) {
                    s.time = s._lastSimTimeWhileRunning;
                    if (s.clock) s.clock.setTime(s.time, now);
                } else {
                    s.time = nextTime;
                }
                s._lastSimTimeWhileRunning = s.time;
            }

            this.updateSimulationLogic(tick.delta);
            F1Sim.Renderer.draw();
            F1Sim.UI.updateUI();

            if (!s.clock.running && s.running) {
                // Clock stopped naturally (reached end)
                s.running = false;
                s.finished = true;
                if (F1Sim.UI.elements['play-pause-btn']) F1Sim.UI.elements['play-pause-btn'].textContent = '↺ Reiniciar';
                F1Sim.UI.updateUI();
            }
        },

        updateSimulationLogic(delta) {
            const s = F1Sim.State.simState;
            const data = F1Sim.State.simData;
            
            if (!data || !data.drivers || !data.circuit_path || data.circuit_path.length === 0) return;
            
            const path = data.circuit_path;
            const pathLen = path.length;
            s.positions = [];
            
            // Ensure finishedDrivers exists
            if (!s.finishedDrivers) s.finishedDrivers = [];

            Object.values(data.drivers).forEach(driver => {
                const driverId = driver.driver_id;
                const laps = driver.laps; 
                
                if (!laps || laps.length === 0) return;
                
                let currentLap = null;
                let prevCumulative = 0;
                let finished = false;
                let totalTime = 0;
                
                // Get total race time for this driver
                const raceTotalTime = laps[laps.length - 1].cumulative;

                // Find current lap based on time
                if (s.time < laps[0].cumulative) {
                     // First lap or before start
                     if (s.time >= 0) currentLap = laps[0];
                } else {
                    // Check if finished
                    if (s.time >= raceTotalTime) {
                         finished = true;
                         totalTime = raceTotalTime;
                         
                         // Check if just finished (not in list yet)
                         const alreadyFinished = s.finishedDrivers.find(d => d.driverId === driverId);
                         if (!alreadyFinished) {
                             console.log(`[DEBUG] Driver Finished: ${driverId} at ${totalTime.toFixed(3)}s`);
                             s.finishedDrivers.push({ driverId, totalTime });
                         }
                    } else {
                        // Iterate
                        for (let i = 0; i < laps.length; i++) {
                            if (s.time < laps[i].cumulative) {
                                currentLap = laps[i];
                                break;
                            }
                            prevCumulative = laps[i].cumulative;
                        }
                    }
                }
                
                let x = 0, y = 0;
                let currentCompound = 'MEDIUM';

                let lapsCompleted = 0;
                let currentProgress = 0;

                if (finished) {
                    x = path[0].x; // Finish line
                    y = path[0].y;
                    currentCompound = laps[laps.length-1].compound;
                    lapsCompleted = laps[laps.length-1].lap;
                    currentProgress = 1.0;
                } else if (!currentLap) {
                     // Before start
                    x = path[0].x;
                    y = path[0].y;
                    currentCompound = laps[0].compound;
                    lapsCompleted = 0;
                    currentProgress = 0.0;
                } else {
                    const lapStartTime = prevCumulative;
                    const lapDuration = currentLap.time;
                    const timeInLap = s.time - lapStartTime;
                    const progress = Math.max(0, Math.min(1, timeInLap / lapDuration));
                    currentCompound = currentLap.compound;
                    
                    // Map progress to track index
                    const idxFloat = progress * (pathLen - 1);
                    const idx1 = Math.floor(idxFloat);
                    const idx2 = Math.min(idx1 + 1, pathLen - 1);
                    const t = idxFloat - idx1;
                    
                    const p1 = path[idx1];
                    const p2 = path[idx2];
                    
                    x = p1.x + (p2.x - p1.x) * t;
                    y = p1.y + (p2.y - p1.y) * t;

                    lapsCompleted = currentLap.lap - 1;
                    currentProgress = progress;
                }
                
                s.positions.push({
                    driver_id: driverId,
                    x: x,
                    y: y,
                    tyre: currentCompound,
                    lapsCompleted: lapsCompleted,
                    currentProgress: currentProgress,
                    pos: 0,
                    finished: finished,
                    totalTime: finished ? totalTime : 0
                });
            });
            
            // Calculate positions (rankings)
            s.positions.sort((a, b) => {
                // If both finished, sort by totalTime (asc)
                if (a.finished && b.finished) {
                    return a.totalTime - b.totalTime;
                }
                // If one finished, they are ahead
                if (a.finished) return -1;
                if (b.finished) return 1;

                // Both running: sort by laps then progress
                if (a.lapsCompleted !== b.lapsCompleted) {
                    return b.lapsCompleted - a.lapsCompleted;
                }
                return b.currentProgress - a.currentProgress;
            });

            s.positions.forEach((p, i) => {
                p.pos = i + 1;
            });
            
            const refs = s.positionChangeRef || {};
            s.positionChangeRef = refs;
            if (!s.driverById) s.driverById = {};

            s.positions.forEach(p => {
                const driverId = p.driver_id;
                const lapNow = Number.isFinite(p.lapsCompleted) ? p.lapsCompleted : 0;
                const key = String(driverId);
                const currentRef = refs[key];
                const meta = s.driverById[key] || null;
                
                if (!currentRef) {
                    let basePos = p.pos;
                    if (meta && Number.isFinite(meta.GridPosition)) {
                        basePos = meta.GridPosition;
                    }
                    refs[key] = { lap: lapNow, pos: basePos };
                } else if (lapNow !== currentRef.lap) {
                    refs[key] = { lap: lapNow, pos: p.pos };
                }
            });
            
            // Debug log final order if simulation just finished
            if (s.finished && !s.finalOrderLogged) {
                 console.log("[DEBUG] Simulation Finished. Final Order:");
                 s.positions.forEach(p => console.log(`${p.pos}. ${p.driver_id} (Finished: ${p.finished}, Time: ${p.totalTime})`));
                 
                 // Validation: Compare final table positions with real-time arrival tracking
                 const finishedSorted = [...s.finishedDrivers].sort((a, b) => a.totalTime - b.totalTime);
                 const positionSorted = s.positions.filter(p => p.finished);
                 
                 let valid = true;
                 if (finishedSorted.length !== positionSorted.length) {
                     console.error("[VALIDATION ERROR] Count mismatch: Real-time finished count", finishedSorted.length, "vs Final Table count", positionSorted.length);
                     valid = false;
                 } else {
                     for (let i = 0; i < finishedSorted.length; i++) {
                         if (finishedSorted[i].driverId !== positionSorted[i].driver_id) {
                             console.error(`[VALIDATION ERROR] Order mismatch at pos ${i+1}: Real-time=${finishedSorted[i].driverId} vs Table=${positionSorted[i].driver_id}`);
                             valid = false;
                         }
                     }
                 }
                 
                 if (valid) {
                     console.log("[VALIDATION SUCCESS] Final table matches real-time finish tracking perfectly.");
                 } else {
                     console.warn("[VALIDATION WARNING] Discrepancies found between real-time tracking and final table sort.");
                 }

                 s.finalOrderLogged = true;
            } else if (!s.finished) {
                s.finalOrderLogged = false;
            }
        }
    }
};

// ==========================================
// 3. Initialization
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
    F1Sim.UI.init();
});
