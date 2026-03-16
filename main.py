from brain import Brain

brain = Brain(alpha=0.2)

print(brain.summary())
print()

for t in range(20):
    sensory = brain.build_sensory_vector({
        "S_left": 1.0,
        "S_right": 0.2,
    })

    x, total_input = brain.step(sensory)

    print(
        f"t={t:02d} | "
        f"M_left={brain.get_activity_by_name('M_left'): .3f} | "
        f"M_right={brain.get_activity_by_name('M_right'): .3f}"
    )