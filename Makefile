# =============================================================================
# Franka Panda Robotics — root Makefile
# =============================================================================
# Targets
# -------
#   make build        Build the Task 3 C++ binary (CMake, Release mode)
#   make lint         Run pylint over all Python packages
#   make test         Run all Python validation scripts (headless)
#   make test-task2   Run Task 2 planner validation only
#   make test-task3   Run Task 3 PID validation only
#   make test-task4   Run Task 4 moteus validation only
#   make all          build + lint + test
#   make clean        Remove build artifacts and Python caches
#
# Prerequisites
# -------------
#   C++:    cmake >= 3.16, g++ with C++17 support
#   Python: pip install -r requirements.txt
# =============================================================================

.PHONY: all build lint test test-task2 test-task3 test-task4 clean

PYTHON    := python3
BUILD_DIR := task3_haptic_pid/build

# ── Default target ─────────────────────────────────────────────────────────
all: build lint test

# ── C++ build (Task 3) ──────────────────────────────────────────────────────
build:
	@echo "── Building Task 3 C++ (CMake) ──────────────────────────────"
	cmake -S task3_haptic_pid -B $(BUILD_DIR) \
	      -DCMAKE_BUILD_TYPE=Release \
	      --no-warn-unused-cli -Wno-dev 2>&1 | tail -3
	cmake --build $(BUILD_DIR) --parallel
	@echo "  Binary → $(BUILD_DIR)/haptic_pid_demo"

# ── Python linting ──────────────────────────────────────────────────────────
lint:
	@echo "── pylint ───────────────────────────────────────────────────"
	$(PYTHON) -m pylint \
	    task1_perception/ \
	    task2_motion_planning/apf_rrt_planner.py \
	    task4_moteus/moteus_actuator.py \
	    tests/test_task2_planner.py \
	    tests/test_task3_pid.py \
	    tests/test_task4_moteus.py \
	    --rcfile=.pylintrc

# ── Tests ────────────────────────────────────────────────────────────────────
test: test-task2 test-task3 test-task4

test-task2:
	@echo "── Test: Task 2 planner validation ──────────────────────────"
	$(PYTHON) tests/test_task2_planner.py

test-task3:
	@echo "── Test: Task 3 PID validation ──────────────────────────────"
	$(PYTHON) tests/test_task3_pid.py

test-task4:
	@echo "── Test: Task 4 moteus validation ───────────────────────────"
	$(PYTHON) tests/test_task4_moteus.py

# ── Clean ────────────────────────────────────────────────────────────────────
clean:
	@echo "── Cleaning build artifacts ─────────────────────────────────"
	rm -rf $(BUILD_DIR)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "  Done."
