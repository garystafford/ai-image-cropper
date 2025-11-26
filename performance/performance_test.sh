START_TIME=$(date +%s.%N)
SOURCE_DIRS=('250_vehicles_open')
METHODS=('yolo' 'detr' 'rt-detr')
CONFIDENCE=0.50
PADDING=5

output=""

for dir in "${SOURCE_DIRS[@]}"; do
    for method in "${METHODS[@]}"; do
        rm -rf objects
        for image in "$dir"/*.jpg; do
            uv run crop-cli "$image" --method "$method" --batch-crop \
                --padding "$PADDING" --confidence "$CONFIDENCE" \
                --batch-output-dir objects
        done
        
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
        SOURCE_COUNT=$(ls "$dir"/*.jpg | wc -l)
        OBJECT_COUNT=$(find objects -type f | wc -l)

        output+="\n"
        output+="\n------------------------------------------------"
        output+="\nPerformance Test Results: $dir | $method"
        output+="\n------------------------------------------------"
        output+="\nSource directory: $dir"
        output+="\nSource image count: $SOURCE_COUNT"
        output+="\nConfiguration: METHOD=$method, CONFIDENCE=$CONFIDENCE, PADDING=$PADDING"
        output+="\nScript run time: $ELAPSED seconds"
        output+="\nTotal cropped objects: $OBJECT_COUNT"
        output+="\nImage per minute: $(echo "scale=2; (60 / $ELAPSED) * $SOURCE_COUNT" | bc)"
        
        echo -fn "$output"
    done
done

echo -fn "$output"