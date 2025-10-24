#!/bin/bash

# DFBI Examples Runner Script (Bash)
# ==================================
# 
# This script provides an interactive way to run DFBI examples
# and explore the library's capabilities.

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo ""
    echo -e "${YELLOW}============================================================${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}============================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if Python is available
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        local version=$(python3 --version 2>&1)
        print_success "Python found: $version"
        return 0
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        local version=$(python --version 2>&1)
        print_success "Python found: $version"
        return 0
    else
        print_error "Python not found. Please install Python 3.8+ and add it to PATH."
        return 1
    fi
}

# Check if DFBI package is installed
check_dfbi_package() {
    if $PYTHON_CMD -c "import dfbi; print('DFBI version:', getattr(dfbi, '__version__', 'unknown'))" 2>/dev/null; then
        print_success "DFBI package is available"
        return 0
    else
        print_error "DFBI package not found. Installing from local directory..."
        return 1
    fi
}

# Install DFBI package
install_dfbi() {
    print_info "Installing DFBI package..."
    
    if [ -d "dfbi_lib_0_1_6_wave_kernels" ]; then
        cd dfbi_lib_0_1_6_wave_kernels
        $PYTHON_CMD -m pip install -e . || {
            print_error "Failed to install DFBI package"
            cd ..
            return 1
        }
        cd ..
        print_success "DFBI package installed successfully"
        return 0
    else
        print_error "DFBI source directory not found"
        return 1
    fi
}

# Install required dependencies
install_dependencies() {
    print_info "Installing required dependencies..."
    
    $PYTHON_CMD -m pip install numpy pandas pyyaml psutil matplotlib scikit-learn || {
        print_warning "Some dependencies may have failed to install"
        return 1
    }
    
    print_success "Dependencies installed successfully"
    return 0
}

# Run specific example
run_example() {
    local example_path="$1"
    local example_name="$2"
    
    print_header "Running $example_name"
    
    if [ -f "$example_path" ]; then
        local example_dir=$(dirname "$example_path")
        local example_file=$(basename "$example_path")
        
        cd "$example_dir"
        
        if $PYTHON_CMD "$example_file"; then
            cd - > /dev/null
            print_success "$example_name completed successfully"
            return 0
        else
            cd - > /dev/null
            print_error "$example_name failed"
            return 1
        fi
    else
        print_error "Example file not found: $example_path"
        return 1
    fi
}

# Show help
show_help() {
    print_header "DFBI Examples Runner - Help"
    
    echo "Usage:"
    echo "  ./run_examples.sh [options]"
    echo ""
    echo "Options:"
    echo "  -e, --example <name>   Run specific example (1-6 or name)"
    echo "  -a, --all             Run all examples"
    echo "  -t, --test            Run tests"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Available Examples:"
    echo "  1, basic              Basic text analysis"
    echo "  2, authorship         Authorship attribution"
    echo "  3, language           Language detection"
    echo "  4, wavelet            Wavelet analysis"
    echo "  5, batch              Batch processing"
    echo "  6, performance        Performance optimization"
    echo ""
    echo "Examples:"
    echo "  ./run_examples.sh -e 1"
    echo "  ./run_examples.sh --example basic"
    echo "  ./run_examples.sh --all"
    echo ""
}

# Interactive menu
show_interactive_menu() {
    print_header "DFBI Examples - Interactive Menu"
    
    echo "Available Examples:"
    echo "  1. Basic Analysis          - Simple fingerprinting and comparison"
    echo "  2. Authorship Attribution - Author identification techniques"
    echo "  3. Language Detection      - Multi-language text analysis"
    echo "  4. Wavelet Analysis        - Advanced kernel usage"
    echo "  5. Batch Processing        - Large-scale text processing"
    echo "  6. Performance Optimization - Benchmarking and tuning"
    echo "  A. Run All Examples"
    echo "  T. Run Tests"
    echo "  Q. Quit"
    echo ""
    
    while true; do
        read -p "Select an option (1-6, A, T, Q): " choice
        
        case $choice in
            1)
                run_example "examples/01_basic_analysis/example.py" "Basic Analysis"
                ;;
            2)
                run_example "examples/02_authorship_attribution/example.py" "Authorship Attribution"
                ;;
            3)
                run_example "examples/03_language_detection/example.py" "Language Detection"
                ;;
            4)
                run_example "examples/04_wavelet_analysis/example.py" "Wavelet Analysis"
                ;;
            5)
                run_example "examples/05_batch_processing/example.py" "Batch Processing"
                ;;
            6)
                run_example "examples/06_performance_optimization/example.py" "Performance Optimization"
                ;;
            [Aa])
                run_all_examples
                ;;
            [Tt])
                run_tests
                ;;
            [Qq])
                print_info "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid choice. Please select 1-6, A, T, or Q."
                continue
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue or Q to quit: " continue_choice
        if [[ $continue_choice =~ ^[Qq]$ ]]; then
            break
        fi
    done
}

# Run all examples
run_all_examples() {
    print_header "Running All Examples"
    
    local examples=(
        "examples/01_basic_analysis/example.py:Basic Analysis"
        "examples/02_authorship_attribution/example.py:Authorship Attribution"
        "examples/03_language_detection/example.py:Language Detection"
        "examples/04_wavelet_analysis/example.py:Wavelet Analysis"
        "examples/05_batch_processing/example.py:Batch Processing"
        "examples/06_performance_optimization/example.py:Performance Optimization"
    )
    
    local success_count=0
    local total_count=${#examples[@]}
    
    for example in "${examples[@]}"; do
        IFS=':' read -r example_path example_name <<< "$example"
        if run_example "$example_path" "$example_name"; then
            ((success_count++))
        fi
        echo ""
    done
    
    print_header "Summary"
    echo "Completed: $success_count / $total_count examples"
    
    if [ $success_count -eq $total_count ]; then
        print_success "All examples completed successfully!"
    else
        print_error "Some examples failed. Check the output above for details."
    fi
}

# Run tests
run_tests() {
    print_header "Running Tests"
    
    if [ -d "dfbi_lib_0_1_6_wave_kernels/tests" ]; then
        cd dfbi_lib_0_1_6_wave_kernels
        
        if $PYTHON_CMD -m pytest tests/ -v; then
            cd - > /dev/null
            print_success "All tests passed!"
        else
            cd - > /dev/null
            print_error "Some tests failed."
        fi
    else
        print_error "Tests directory not found"
    fi
}

# Main execution
main() {
    print_header "DFBI Examples Runner"
    
    # Check prerequisites
    if ! check_python; then
        exit 1
    fi
    
    if ! check_dfbi_package; then
        if ! install_dfbi; then
            exit 1
        fi
    fi
    
    # Install dependencies
    install_dependencies || print_warning "Some dependencies may be missing"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--example)
                EXAMPLE="$2"
                shift 2
                ;;
            -a|--all)
                ALL=true
                shift
                ;;
            -t|--test)
                TEST=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Handle command line arguments
    if [ "$ALL" = true ]; then
        run_all_examples
        exit 0
    fi
    
    if [ "$TEST" = true ]; then
        run_tests
        exit 0
    fi
    
    if [ -n "$EXAMPLE" ]; then
        case $EXAMPLE in
            1|basic)
                run_example "examples/01_basic_analysis/example.py" "Basic Analysis"
                ;;
            2|authorship)
                run_example "examples/02_authorship_attribution/example.py" "Authorship Attribution"
                ;;
            3|language)
                run_example "examples/03_language_detection/example.py" "Language Detection"
                ;;
            4|wavelet)
                run_example "examples/04_wavelet_analysis/example.py" "Wavelet Analysis"
                ;;
            5|batch)
                run_example "examples/05_batch_processing/example.py" "Batch Processing"
                ;;
            6|performance)
                run_example "examples/06_performance_optimization/example.py" "Performance Optimization"
                ;;
            *)
                print_error "Unknown example: $EXAMPLE"
                show_help
                exit 1
                ;;
        esac
        exit 0
    fi
    
    # Show interactive menu if no arguments provided
    show_interactive_menu
}

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Run main function with all arguments
main "$@"