import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd

# -------------------------------------------------------------
# PAGE CONFIGURATION & DARK THEME CUSTOM CSS
# -------------------------------------------------------------
st.set_page_config(
    page_title="Dual-Bogie Digital Twin Telemetry",
    page_icon="⚙️",
    layout="wide"
)

# Custom CSS for UI Matching
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0b0f17;
        color: #e0e6ed;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d131d;
        border-right: 1px solid #1a2332;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #8b9bb4;
    }
    
    /* Custom Card Containers for Metrics */
    .metric-card {
        background-color: #111823;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: left;
    }
    
    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Custom Alert Banner */
    .status-banner {
        border-radius: 8px;
        padding: 10px 16px;
        text-align: center;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 1. DIRECT MODEL LOADING
# -------------------------------------------------------------
@st.cache_resource
def load_ml_artifacts():
    try:
        model_kinematic = joblib.load("models/axle_lock_xgb.joblib")
        transformer_kinematic = joblib.load("models/power_transformer.joblib")
        model_phy = joblib.load("models/phy_axle_lock_xgb.joblib")
        transformer_phy = joblib.load("models/phy_power_transformer.joblib")
        return model_kinematic, transformer_kinematic, model_phy, transformer_phy, None
    except Exception as e:
        return None, None, None, None, str(e)

model_kin, trans_kin, model_phy, trans_phy, load_error = load_ml_artifacts()

# -------------------------------------------------------------
# 2. SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.caption("Live telemetry inputs for all four axles. Every change streams to the inference engine and updates the twin in real time.")

# 1. Kinematics
with st.sidebar.expander("1. Train Kinematics", expanded=True):
    v_loco_kmh = st.slider("Locomotive Speed", 0.0, 160.0, 80.0, 1.0, format="%.0f km/h")
    axle1_speed = st.slider("Axle 1 Speed", 0.0, 200.0, 55.1, 0.5, format="%.1f rad/s")
    axle2_speed = st.slider("Axle 2 Speed", 0.0, 200.0, 55.2, 0.5, format="%.1f rad/s")
    axle3_speed = st.slider("Axle 3 Speed", 0.0, 200.0, 55.0, 0.5, format="%.1f rad/s")
    axle4_speed = st.slider("Axle 4 Speed", 0.0, 200.0, 54.8, 0.5, format="%.1f rad/s")
    axle1_slip = st.slider("Axle 1 Slip Ratio", 0.0, 1.0, 0.01, 0.01)

# 2. Axle 1
with st.sidebar.expander("2. Axle 1 – Front Bogie", expanded=False):
    axle1_temp = st.slider("Bearing Temp", 20.0, 150.0, 150.0, format="%.1f °C")
    axle1_vib = st.slider("Vibration", 0.0, 10.0, 9.75, format="%.3f G")
    axle1_amp = st.slider("Motor Current", 0.0, 700.0, 300.0, format="%.0f A")

# 3. Axle 2
with st.sidebar.expander("3. Axle 2 – Front Bogie", expanded=False):
    axle2_temp = st.slider("Bearing Temp ", 20.0, 150.0, 150.0, format="%.1f °C")
    axle2_vib = st.slider("Vibration ", 0.0, 10.0, 0.306, format="%.3f G")
    axle2_amp = st.slider("Motor Current ", 0.0, 700.0, 300.0, format="%.0f A")

# 4. Axle 3
with st.sidebar.expander("4. Axle 3 – Rear Bogie", expanded=False):
    axle3_temp = st.slider("Bearing Temp  ", 20.0, 150.0, 46.2, format="%.1f °C")
    axle3_vib = st.slider("Vibration  ", 0.0, 10.0, 0.356, format="%.3f G")
    axle3_amp = st.slider("Motor Current  ", 0.0, 700.0, 305.0, format="%.0f A")

# 5. Axle 4
with st.sidebar.expander("5. Axle 4 – Rear Bogie", expanded=False):
    axle4_temp = st.slider("Bearing Temp   ", 20.0, 150.0, 44.8, format="%.1f °C")
    axle4_vib = st.slider("Vibration   ", 0.0, 10.0, 0.286, format="%.3f G")
    axle4_amp = st.slider("Motor Current   ", 0.0, 700.0, 298.0, format="%.0f A")

# -------------------------------------------------------------
# 3. LOCAL INFERENCE ENGINE
# -------------------------------------------------------------
lock_threshold = 5.0 if v_loco_kmh > 15.0 else -1.0
is_locked1 = 1 if axle1_speed < lock_threshold else 0
is_locked2 = 1 if axle2_speed < lock_threshold else 0
is_locked3 = 1 if axle3_speed < lock_threshold else 0
is_locked4 = 1 if axle4_speed < lock_threshold else 0

pred_kin, prob_kin = 0, 0.0
pred_phy, prob_phy = 0, 0.0

if model_kin and model_phy:
    try:
        df_kinematic = pd.DataFrame([{
            "v_loco_kmh": float(v_loco_kmh),
            "axle1_speed_rads": float(axle1_speed),
            "axle2_speed_rads": float(axle2_speed),
            "axle3_speed_rads": float(axle3_speed),
            "axle4_speed_rads": float(axle4_speed),
            "axle1_slip_ratio": float(axle1_slip)
        }])

        df_phy = pd.DataFrame([{
            "axle1_bearing_temp_c": float(axle1_temp),
            "axle1_vibration_g": float(axle1_vib),
            "axle1_motor_current_amp": float(axle1_amp),
            "axle2_bearing_temp_c": float(axle2_temp),
            "axle2_vibration_g": float(axle2_vib),
            "axle2_motor_current_amp": float(axle2_amp),
            "axle3_bearing_temp_c": float(axle3_temp),
            "axle3_vibration_g": float(axle3_vib),
            "axle3_motor_current_amp": float(axle3_amp),
            "axle4_bearing_temp_c": float(axle4_temp),
            "axle4_vibration_g": float(axle4_vib),
            "axle4_motor_current_amp": float(axle4_amp)
        }])

        x_scaled_kin = trans_kin.transform(df_kinematic)
        x_scaled_phy = trans_phy.transform(df_phy)

        pred_kin = int(model_kin.predict(x_scaled_kin)[0])
        prob_kin = float(model_kin.predict_proba(x_scaled_kin)[0][1])

        pred_phy = int(model_phy.predict(x_scaled_phy)[0])
        prob_phy = float(model_phy.predict_proba(x_scaled_phy)[0][1])
    except Exception:
        pass

# Rules Override
speeds = [axle1_speed, axle2_speed, axle3_speed, axle4_speed]
if v_loco_kmh > 15.0 and any(s < 5.0 for s in speeds):
    pred_kin = 1
    prob_kin = max(prob_kin, 0.99)

# Alert Status
if pred_kin == 1 and pred_phy == 1:
    alert_status = "SYSTEM CRITICAL — AXLE LOCK & MECHANICAL SEIZURE DETECTED"
    banner_bg, banner_text_col = "rgba(220, 38, 38, 0.2)", "#f87171"
    risk_level = "CRITICAL"
elif pred_phy == 1:
    alert_status = "WARNING — BEARING TEMPERATURE / VIBRATION HAZARD"
    banner_bg, banner_text_col = "rgba(234, 88, 12, 0.2)", "#fb923c"
    risk_level = "WARNING"
elif pred_kin == 1:
    alert_status = "CAUTION — KINEMATIC LOCK DETECTED"
    banner_bg, banner_text_col = "rgba(202, 138, 4, 0.2)", "#facc15"
    risk_level = "ELEVATED"
else:
    alert_status = "SYSTEM NORMAL (LOCAL PREVIEW — ACTIVE)"
    banner_bg, banner_text_col = "rgba(16, 185, 129, 0.15)", "#34d399"
    risk_level = "NORMAL"

# -------------------------------------------------------------
# 4. TOP DISPLAY BANNER & CARDS
# -------------------------------------------------------------
st.markdown(
    f"""
    <div class="status-banner" style="background-color: {banner_bg}; color: {banner_text_col}; border-color: {banner_text_col}44;">
        ● {alert_status}
    </div>
    """,
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Level</div>
            <div class="metric-value">{risk_level}</div>
        </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Kinematic Anomaly</div>
            <div class="metric-value">{prob_kin:.1%}</div>
        </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Physical Hazard</div>
            <div class="metric-value">{prob_phy:.1%}</div>
        </div>
    """, unsafe_allow_html=True)

st.write("")

# Dynamic color logic
def get_status(temp, vib, is_locked):
    if is_locked or temp > 90 or vib > 2.5:
        return "#ef4444", "ERR"
    elif temp > 70 or vib > 1.2:
        return "#f97316", "WARN"
    else:
        return "#10b981", "OK"

c_a1, s_a1 = get_status(axle1_temp, axle1_vib, is_locked1)
c_a2, s_a2 = get_status(axle2_temp, axle2_vib, is_locked2)
c_a3, s_a3 = get_status(axle3_temp, axle3_vib, is_locked3)
c_a4, s_a4 = get_status(axle4_temp, axle4_vib, is_locked4)

js_c_a1 = c_a1.replace("#", "0x")
js_c_a2 = c_a2.replace("#", "0x")
js_c_a3 = c_a3.replace("#", "0x")
js_c_a4 = c_a4.replace("#", "0x")

track_speed_factor = float(v_loco_kmh) * 0.003

# -------------------------------------------------------------
# 5. THREE.JS VIEWPORT WITH OVERLAY BADGES
# -------------------------------------------------------------
three_js_code = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ margin: 0; overflow: hidden; background: #070a0e; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
        canvas {{ width: 100vw; height: 100vh; display: block; }}
        
        /* Top Overlay Badges matched to reference design */
        #axle-bar {{
            position: absolute;
            top: 14px;
            left: 14px;
            right: 14px;
            display: flex;
            gap: 10px;
            z-index: 10;
            pointer-events: none;
        }}
        
        .axle-pill {{
            background: rgba(13, 19, 29, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 6px;
            padding: 6px 12px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            color: #94a3b8;
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        
        .dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
        }}
        
        .val {{
            color: #f1f5f9;
            font-weight: 600;
            font-family: monospace;
        }}
        
        #controls-hint {{
            position: absolute;
            bottom: 12px;
            right: 14px;
            font-size: 10px;
            color: #475569;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="axle-bar">
        <div class="axle-pill">
            <span class="dot" style="background: {c_a1};"></span>
            <span>Axle 1</span>
            <span class="val">{axle1_temp:.1f}°C</span>
            <span class="val">{axle1_vib:.3f}G</span>
            <span style="color: {c_a1}; font-weight: bold;">{s_a1}</span>
        </div>
        <div class="axle-pill">
            <span class="dot" style="background: {c_a2};"></span>
            <span>Axle 2</span>
            <span class="val">{axle2_temp:.1f}°C</span>
            <span class="val">{axle2_vib:.3f}G</span>
            <span style="color: {c_a2}; font-weight: bold;">{s_a2}</span>
        </div>
        <div class="axle-pill">
            <span class="dot" style="background: {c_a3};"></span>
            <span>Axle 3</span>
            <span class="val">{axle3_temp:.1f}°C</span>
            <span class="val">{axle3_vib:.3f}G</span>
            <span style="color: {c_a3}; font-weight: bold;">{s_a3}</span>
        </div>
        <div class="axle-pill">
            <span class="dot" style="background: {c_a4};"></span>
            <span>Axle 4</span>
            <span class="val">{axle4_temp:.1f}°C</span>
            <span class="val">{axle4_vib:.3f}G</span>
            <span style="color: {c_a4}; font-weight: bold;">{s_a4}</span>
        </div>
    </div>

    <div id="controls-hint">DRAG TO ROTATE · SCROLL / PINCH TO ZOOM</div>

    <script>
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x070a0e, 0.015);

        const camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(13, 7, 13);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.maxPolarAngle = Math.PI / 2 - 0.02;

        // Grid Floor
        const gridHelper = new THREE.GridHelper(60, 60, 0x1e293b, 0x0f172a);
        gridHelper.position.y = -0.25;
        scene.add(gridHelper);

        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const sun = new THREE.DirectionalLight(0xffffff, 1.8);
        sun.position.set(12, 20, 10);
        sun.castShadow = true;
        scene.add(sun);

        const darkChassisMat = new THREE.MeshStandardMaterial({{ color: 0x1e293b, metalness: 0.8, roughness: 0.2 }});
        const steelReflect = new THREE.MeshStandardMaterial({{ color: 0x94a3b8, metalness: 0.9, roughness: 0.15 }});
        const darkMetal = new THREE.MeshStandardMaterial({{ color: 0x0f172a, metalness: 0.7, roughness: 0.4 }});
        const railMat = new THREE.MeshStandardMaterial({{ color: 0x475569, metalness: 0.95, roughness: 0.1 }});
        const sleeperMat = new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.9 }});

        // Rails & Sleepers
        const sleeperGroup = new THREE.Group();
        const sleeperGeo = new THREE.BoxGeometry(2.8, 0.14, 0.38);

        for (let z = -25; z <= 25; z += 0.85) {{
            const sleeper = new THREE.Mesh(sleeperGeo, sleeperMat);
            sleeper.position.set(0, -0.22, z);
            sleeper.receiveShadow = true;
            sleeperGroup.add(sleeper);
        }}
        scene.add(sleeperGroup);

        const railShape = new THREE.Shape();
        railShape.moveTo(-0.06, 0); railShape.lineTo(0.06, 0);
        railShape.lineTo(0.06, 0.03); railShape.lineTo(0.02, 0.08);
        railShape.lineTo(0.03, 0.16); railShape.lineTo(-0.03, 0.16);
        railShape.lineTo(-0.02, 0.08); railShape.lineTo(-0.06, 0.03);

        const railExtrude = new THREE.ExtrudeGeometry(railShape, {{ depth: 52, bevelEnabled: false }});
        const railL = new THREE.Mesh(railExtrude, railMat);
        railL.position.set(-1.1, -0.15, -26);
        scene.add(railL);

        const railR = railL.clone();
        railR.position.set(1.1, -0.15, -26);
        scene.add(railR);

        // Center Spine Chassis
        const centerSpineGeo = new THREE.BoxGeometry(1.2, 0.22, 9.0);
        const centerSpine = new THREE.Mesh(centerSpineGeo, darkChassisMat);
        centerSpine.position.set(0, 0.5, 0);
        centerSpine.castShadow = true;
        scene.add(centerSpine);

        function createBogieBlock(centerZ) {{
            const bogie = new THREE.Group();

            const transomGeo = new THREE.BoxGeometry(2.2, 0.24, 0.85);
            const transom = new THREE.Mesh(transomGeo, darkChassisMat);
            transom.position.set(0, 0.4, 0);
            transom.castShadow = true;
            bogie.add(transom);

            const sideBeamGeo = new THREE.BoxGeometry(0.18, 0.26, 2.7);
            const sideL = new THREE.Mesh(sideBeamGeo, darkChassisMat);
            sideL.position.set(-1.22, 0.3, 0);
            sideL.castShadow = true;
            bogie.add(sideL);

            const sideR = sideL.clone();
            sideR.position.set(1.22, 0.3, 0);
            bogie.add(sideR);

            bogie.position.set(0, 0, centerZ);
            scene.add(bogie);
        }}

        createBogieBlock(-3.35);
        createBogieBlock(3.35);

        function createAxle(colorHex, isLocked, tempVal, vibVal, ampVal, speedVal, posZ) {{
            const group = new THREE.Group();

            const axleGeo = new THREE.CylinderGeometry(0.07, 0.07, 2.3, 32);
            axleGeo.rotateZ(Math.PI / 2);
            const axleMesh = new THREE.Mesh(axleGeo, steelReflect);
            group.add(axleMesh);

            const wheelGeo = new THREE.CylinderGeometry(0.48, 0.48, 0.12, 32);
            wheelGeo.rotateZ(Math.PI / 2);

            const wL = new THREE.Mesh(wheelGeo, steelReflect);
            wL.position.set(-1.1, 0, 0);
            group.add(wL);

            const wR = new THREE.Mesh(wheelGeo, steelReflect);
            wR.position.set(1.1, 0, 0);
            group.add(wR);

            const boxGeo = new THREE.BoxGeometry(0.26, 0.28, 0.26);
            const boxMat = new THREE.MeshStandardMaterial({{
                color: parseInt(colorHex),
                emissive: parseInt(colorHex),
                emissiveIntensity: (tempVal > 70 || isLocked == 1) ? 0.9 : 0.1,
                roughness: 0.2
            }});

            const boxL = new THREE.Mesh(boxGeo, boxMat);
            boxL.position.set(-1.28, 0, 0);
            group.add(boxL);

            const boxR = new THREE.Mesh(boxGeo, boxMat);
            boxR.position.set(1.28, 0, 0);
            group.add(boxR);

            group.position.set(0, 0.26, posZ);
            scene.add(group);

            let particles = null;
            if (tempVal > 70) {{
                const pCount = 20;
                const pGeo = new THREE.BufferGeometry();
                const pPos = new Float32Array(pCount * 3);
                for(let i = 0; i < pCount * 3; i += 3) {{
                    pPos[i] = (Math.random() - 0.5) * 2.5;
                    pPos[i+1] = 0.3 + Math.random() * 0.4;
                    pPos[i+2] = posZ + (Math.random() - 0.5) * 0.2;
                }}
                pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
                const pMat = new THREE.PointsMaterial({{ color: 0xef4444, size: 0.08, transparent: true, opacity: 0.8 }});
                particles = new THREE.Points(pGeo, pMat);
                scene.add(particles);
            }}

            return {{ group, wL, wR, isLocked, vibVal, particles }};
        }}

        const axle1 = createAxle("{js_c_a1}", {is_locked1}, {axle1_temp}, {axle1_vib}, {axle1_amp}, {axle1_speed}, -4.5);
        const axle2 = createAxle("{js_c_a2}", {is_locked2}, {axle2_temp}, {axle2_vib}, {axle2_amp}, {axle2_speed}, -2.2);
        const axle3 = createAxle("{js_c_a3}", {is_locked3}, {axle3_temp}, {axle3_vib}, {axle3_amp}, {axle3_speed}, 2.2);
        const axle4 = createAxle("{js_c_a4}", {is_locked4}, {axle4_temp}, {axle4_vib}, {axle4_amp}, {axle4_speed}, 4.5);

        const axlesList = [axle1, axle2, axle3, axle4];
        const trackSpeed = {track_speed_factor};
        const speeds = [
            {float(axle1_speed) * 0.0012},
            {float(axle2_speed) * 0.0012},
            {float(axle3_speed) * 0.0012},
            {float(axle4_speed) * 0.0012}
        ];

        function animate() {{
            requestAnimationFrame(animate);

            sleeperGroup.children.forEach(s => {{
                s.position.z += trackSpeed;
                if (s.position.z > 25) s.position.z -= 50;
            }});

            axlesList.forEach((ax, idx) => {{
                if (!ax.isLocked) {{
                    ax.wL.rotation.x += speeds[idx];
                    ax.wR.rotation.x += speeds[idx];
                }}

                if (ax.vibVal > 0.5 || ax.isLocked) {{
                    const shake = (ax.isLocked ? 1.5 : 1.0) * ax.vibVal * 0.005;
                    ax.group.position.x = (Math.random() - 0.5) * shake;
                    ax.group.position.y = 0.26 + (Math.random() - 0.5) * shake;
                }} else {{
                    ax.group.position.x = 0;
                    ax.group.position.y = 0.26;
                }}

                if (ax.particles) {{
                    const pos = ax.particles.geometry.attributes.position.array;
                    for(let i=1; i<pos.length; i+=3) {{
                        pos[i] += 0.008;
                        if (pos[i] > 1.2) pos[i] = 0.3;
                    }}
                    ax.particles.geometry.attributes.position.needsUpdate = true;
                }}
            }});

            controls.update();
            renderer.render(scene, camera);
        }}

        animate();

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
"""

components.html(three_js_code, height=540)