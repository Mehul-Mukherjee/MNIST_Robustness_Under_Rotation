import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("🧠 MNIST Robustness Under Rotation")

# -----------------------
# REAL RESULTS (from your evaluation)
# -----------------------
angles = [0, 15, 30, 45, 60, 90]

clean_acc = [0.988, 0.965, 0.842, 0.556, 0.287, 0.163]
aug_acc   = [0.984, 0.979, 0.966, 0.886, 0.648, 0.185]

# -----------------------
# UI Section
# -----------------------
st.subheader("📊 Accuracy vs Rotation Angle")

fig, ax = plt.subplots()

ax.plot(angles, clean_acc, marker="o", label="Clean CNN")
ax.plot(angles, aug_acc, marker="o", label="Augmented CNN (Rotation + Noise)")

ax.set_xlabel("Rotation Angle (degrees)")
ax.set_ylabel("Accuracy")
ax.set_title("Model Robustness Comparison")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# -----------------------
# INSIGHT SECTION
# -----------------------
st.subheader("🧠 Key Insight")

st.write("""
- Clean CNN collapses under rotation (distribution shift sensitivity)
- Augmented CNN generalizes significantly better
- Data augmentation improves rotational invariance
""")

# -----------------------
# METRICS HIGHLIGHT
# -----------------------
st.subheader("📉 Performance Drop (0° → 90°)")

clean_drop = clean_acc[0] - clean_acc[-1]
aug_drop = aug_acc[0] - aug_acc[-1]

st.metric("Clean Model Drop", f"{clean_drop:.3f}")
st.metric("Augmented Model Drop", f"{aug_drop:.3f}")