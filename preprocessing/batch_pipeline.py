    # ------------------------------------------------
    # HD-BET skull stripping
    # ------------------------------------------------

    skull_dir = (
        subject_dir / "skull_stripped"
    )

    skull_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    brain_output = (
        skull_dir /
        f"{subject}_T1w_brain.nii.gz"
    )

    print("\n[3/5] Running HD-BET skull stripping...")

    command = [
        "hd-bet",
        "-i",
        str(n4_output),
        "-o",
        str(brain_output),
        "-device",
        "cpu",
        "--save_bet_mask"
    ]

    result = subprocess.run(command)

    if result.returncode != 0:

        print(
            f"❌ HD-BET failed: {subject}"
        )

        failed.append(subject)

        continue

    print(
        f"✅ HD-BET complete: {subject}"
    )