$START_TIME = Get-Date
$SOURCE_DIRS = @('250_vehicles_open')
$METHODS = @('rt-detr')
$CONFIDENCE = 0.50
$PADDING = 5

# Check GPU availability
Write-Host "`n================================================"
Write-Host "GPU Detection Check"
Write-Host "================================================"
.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
Write-Host "================================================`n"

$output = ""

foreach ($dir in $SOURCE_DIRS) {
    foreach ($method in $METHODS) {
        # Remove objects directory if it exists
        if (Test-Path "objects") {
            Remove-Item -Recurse -Force "objects"
        }
        
        # Process each image
        $images = Get-ChildItem "$dir\*.jpg"
        $imageCount = 0
        foreach ($image in $images) {
            # Set logging to INFO for first image to see GPU message
            $env:PYTHONUNBUFFERED = "1"
            if ($imageCount -eq 0) {
                Write-Host "`nProcessing first image with detailed logging..." -ForegroundColor Yellow
                $escapedPath = $image.FullName -replace '\\', '\\\\'
                .venv\Scripts\python.exe -c "import logging; logging.basicConfig(level=logging.INFO); from cropper import ImageCropper; c = ImageCropper(r'$($image.FullName)'); c.load_image(); c.find_all_objects_rtdetr(None, $CONFIDENCE)"
                Write-Host ""
            }
            
            .venv\Scripts\python.exe -m cropper "$($image.FullName)" --method "$method" --batch-crop `
                --padding "$PADDING" --confidence "$CONFIDENCE" `
                --batch-output-dir objects
            
            $imageCount++
        }
        
        $END_TIME = Get-Date
        $ELAPSED = ($END_TIME - $START_TIME).TotalSeconds
        $SOURCE_COUNT = (Get-ChildItem "$dir\*.jpg").Count
        $OBJECT_COUNT = (Get-ChildItem "objects" -Recurse -File -ErrorAction SilentlyContinue).Count
        $IMAGES_PER_MINUTE = [math]::Round((60 / $ELAPSED) * $SOURCE_COUNT, 2)

        $output += "`n"
        $output += "`n------------------------------------------------"
        $output += "`nPerformance Test Results: $dir | $method"
        $output += "`n------------------------------------------------"
        $output += "`nSource directory: $dir"
        $output += "`nSource image count: $SOURCE_COUNT"
        $output += "`nConfiguration: METHOD=$method, CONFIDENCE=$CONFIDENCE, PADDING=$PADDING"
        $output += "`nScript run time: $ELAPSED seconds"
        $output += "`nTotal cropped objects: $OBJECT_COUNT"
        $output += "`nImage per minute: $IMAGES_PER_MINUTE"
        
        Write-Host $output
    }
}

Write-Host $output
