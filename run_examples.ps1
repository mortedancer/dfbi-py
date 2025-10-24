# DFBI Examples Runner Script (PowerShell)
# ========================================
# 
# This script provides an interactive way to run DFBI examples
# and explore the library's capabilities.

param(
    [string]$Example = "",
    [switch]$All = $false,
    [switch]$Help = $false
)

# Color functions for better output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    } else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Header($text) {
    Write-Host ""
    Write-ColorOutput Yellow "=" * 60
    Write-ColorOutput Yellow $text
    Write-ColorOutput Yellow "=" * 60
    Write-Host ""
}

function Write-Success($text) {
    Write-ColorOutput Green "✓ $text"
}

function Write-Error($text) {
    Write-ColorOutput Red "✗ $text"
}

function Write-Info($text) {
    Write-ColorOutput Cyan "ℹ $text"
}

# Check if Python is available
function Test-Python {
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Python found: $pythonVersion"
            return $true
        }
    } catch {
        Write-Error "Python not found. Please install Python 3.8+ and add it to PATH."
        return $false
    }
    return $false
}

# Check if DFBI package is installed
function Test-DFBIPackage {
    try {
        python -c "import dfbi; print('DFBI version:', dfbi.__version__ if hasattr(dfbi, '__version__') else 'unknown')" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "DFBI package is available"
            return $true
        }
    } catch {
        Write-Error "DFBI package not found. Installing from local directory..."
        return $false
    }
    return $false
}

# Install DFBI package
function Install-DFBI {
    Write-Info "Installing DFBI package..."
    try {
        Set-Location "dfbi_lib_0_1_6_wave_kernels"
        python -m pip install -e .
        Set-Location ".."
        if ($LASTEXITCODE -eq 0) {
            Write-Success "DFBI package installed successfully"
            return $true
        }
    } catch {
        Write-Error "Failed to install DFBI package"
        return $false
    }
    return $false
}

# Install required dependencies
function Install-Dependencies {
    Write-Info "Installing required dependencies..."
    try {
        python -m pip install numpy pandas pyyaml psutil matplotlib scikit-learn
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Dependencies installed successfully"
            return $true
        }
    } catch {
        Write-Error "Failed to install dependencies"
        return $false
    }
    return $false
}

# Run specific example
function Run-Example($examplePath, $exampleName) {
    Write-Header "Running $exampleName"
    
    if (Test-Path $examplePath) {
        try {
            Set-Location (Split-Path $examplePath -Parent)
            python (Split-Path $examplePath -Leaf)
            Set-Location $PSScriptRoot
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "$exampleName completed successfully"
                return $true
            } else {
                Write-Error "$exampleName failed with exit code $LASTEXITCODE"
                return $false
            }
        } catch {
            Write-Error "Error running $exampleName : $_"
            Set-Location $PSScriptRoot
            return $false
        }
    } else {
        Write-Error "Example file not found: $examplePath"
        return $false
    }
}

# Show help
function Show-Help {
    Write-Header "DFBI Examples Runner - Help"
    
    Write-Host "Usage:"
    Write-Host "  .\run_examples.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Example <name>    Run specific example (1-7 or name)"
    Write-Host "  -All              Run all examples"
    Write-Host "  -Help             Show this help message"
    Write-Host ""
    Write-Host "Available Examples:"
    Write-Host "  1, basic          Basic text analysis"
    Write-Host "  2, authorship     Authorship attribution"
    Write-Host "  3, language       Language detection"
    Write-Host "  4, wavelet        Wavelet analysis"
    Write-Host "  5, batch          Batch processing"
    Write-Host "  6, performance    Performance optimization"
    Write-Host "  7, blog           Advanced blog authorship analysis"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run_examples.ps1 -Example 1"
    Write-Host "  .\run_examples.ps1 -Example basic"
    Write-Host "  .\run_examples.ps1 -All"
    Write-Host ""
}

# Interactive menu
function Show-InteractiveMenu {
    Write-Header "DFBI Examples - Interactive Menu"
    
    Write-Host "Available Examples:"
    Write-Host "  1. Basic Analysis          - Simple fingerprinting and comparison"
    Write-Host "  2. Authorship Attribution - Author identification techniques"
    Write-Host "  3. Language Detection      - Multi-language text analysis"
    Write-Host "  4. Wavelet Analysis        - Advanced kernel usage"
    Write-Host "  5. Batch Processing        - Large-scale text processing"
    Write-Host "  6. Performance Optimization - Benchmarking and tuning"
    Write-Host "  7. Blog Authorship Analysis - Advanced real data analysis"
    Write-Host "  A. Run All Examples"
    Write-Host "  T. Run Tests"
    Write-Host "  Q. Quit"
    Write-Host ""
    
    do {
        $choice = Read-Host "Select an option (1-7, A, T, Q)"
        
        switch ($choice.ToUpper()) {
            "1" { 
                Run-Example "examples\01_basic_analysis\example.py" "Basic Analysis"
                break
            }
            "2" { 
                Run-Example "examples\02_authorship_attribution\example.py" "Authorship Attribution"
                break
            }
            "3" { 
                Run-Example "examples\03_language_detection\example.py" "Language Detection"
                break
            }
            "4" { 
                Run-Example "examples\04_wavelet_analysis\example.py" "Wavelet Analysis"
                break
            }
            "5" { 
                Run-Example "examples\05_batch_processing\example.py" "Batch Processing"
                break
            }
            "6" { 
                Run-Example "examples\06_performance_optimization\example.py" "Performance Optimization"
                break
            }
            "7" { 
                Run-Example "examples\07_blog_authorship_analysis\example.py" "Blog Authorship Analysis"
                break
            }
            "A" { 
                Run-AllExamples
                break
            }
            "T" {
                Run-Tests
                break
            }
            "Q" { 
                Write-Info "Goodbye!"
                return
            }
            default {
                Write-Error "Invalid choice. Please select 1-7, A, T, or Q."
            }
        }
        
        Write-Host ""
        $continue = Read-Host "Press Enter to continue or Q to quit"
        if ($continue.ToUpper() -eq "Q") {
            break
        }
        
    } while ($true)
}

# Run all examples
function Run-AllExamples {
    Write-Header "Running All Examples"
    
    $examples = @(
        @("examples\01_basic_analysis\example.py", "Basic Analysis"),
        @("examples\02_authorship_attribution\example.py", "Authorship Attribution"),
        @("examples\03_language_detection\example.py", "Language Detection"),
        @("examples\04_wavelet_analysis\example.py", "Wavelet Analysis"),
        @("examples\05_batch_processing\example.py", "Batch Processing"),
        @("examples\06_performance_optimization\example.py", "Performance Optimization"),
        @("examples\07_blog_authorship_analysis\example.py", "Blog Authorship Analysis")
    )
    
    $successCount = 0
    $totalCount = $examples.Count
    
    foreach ($example in $examples) {
        if (Run-Example $example[0] $example[1]) {
            $successCount++
        }
        Write-Host ""
    }
    
    Write-Header "Summary"
    Write-Host "Completed: $successCount / $totalCount examples"
    
    if ($successCount -eq $totalCount) {
        Write-Success "All examples completed successfully!"
    } else {
        Write-Error "Some examples failed. Check the output above for details."
    }
}

# Run tests
function Run-Tests {
    Write-Header "Running Tests"
    
    if (Test-Path "dfbi_lib_0_1_6_wave_kernels\tests") {
        try {
            Set-Location "dfbi_lib_0_1_6_wave_kernels"
            python -m pytest tests\ -v
            Set-Location $PSScriptRoot
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "All tests passed!"
            } else {
                Write-Error "Some tests failed."
            }
        } catch {
            Write-Error "Error running tests: $_"
            Set-Location $PSScriptRoot
        }
    } else {
        Write-Error "Tests directory not found"
    }
}

# Main execution
function Main {
    Write-Header "DFBI Examples Runner"
    
    # Check prerequisites
    if (-not (Test-Python)) {
        return
    }
    
    if (-not (Test-DFBIPackage)) {
        if (-not (Install-DFBI)) {
            return
        }
    }
    
    # Install dependencies
    Install-Dependencies | Out-Null
    
    # Handle command line arguments
    if ($Help) {
        Show-Help
        return
    }
    
    if ($All) {
        Run-AllExamples
        return
    }
    
    if ($Example) {
        switch ($Example) {
            { $_ -in @("1", "basic") } {
                Run-Example "examples\01_basic_analysis\example.py" "Basic Analysis"
            }
            { $_ -in @("2", "authorship") } {
                Run-Example "examples\02_authorship_attribution\example.py" "Authorship Attribution"
            }
            { $_ -in @("3", "language") } {
                Run-Example "examples\03_language_detection\example.py" "Language Detection"
            }
            { $_ -in @("4", "wavelet") } {
                Run-Example "examples\04_wavelet_analysis\example.py" "Wavelet Analysis"
            }
            { $_ -in @("5", "batch") } {
                Run-Example "examples\05_batch_processing\example.py" "Batch Processing"
            }
            { $_ -in @("6", "performance") } {
                Run-Example "examples\06_performance_optimization\example.py" "Performance Optimization"
            }
            { $_ -in @("7", "blog") } {
                Run-Example "examples\07_blog_authorship_analysis\example.py" "Blog Authorship Analysis"
            }
            default {
                Write-Error "Unknown example: $Example"
                Show-Help
            }
        }
        return
    }
    
    # Show interactive menu if no arguments provided
    Show-InteractiveMenu
}

# Run main function
Main