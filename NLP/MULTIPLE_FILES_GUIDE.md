# 📚 Multiple Files Upload Guide

## Overview

The Teacher Dashboard now supports **two upload modes**:

### Mode 1: Single Combined File
Upload **one CSV/Excel file** containing all students' data.

**Format:**
```csv
Name,Roll Number,Q1,Q2,Q3
John Doe,101,Answer 1...,Answer 2...,Answer 3...
Jane Smith,102,Answer 1...,Answer 2...,Answer 3...
Bob Johnson,103,Answer 1...,Answer 2...,Answer 3...
```

### Mode 2: Multiple Individual Files ✨ NEW
Upload **multiple CSV/Excel files** (one per student). The system automatically combines them!

**Example:** If each student submits a Google Form individually:

**File 1: john_doe_response.csv**
```csv
Name,Roll Number,Q1,Q2,Q3
John Doe,101,Photosynthesis is...,The water cycle...,DNA contains...
```

**File 2: jane_smith_response.csv**
```csv
Name,Roll Number,Q1,Q2,Q3
Jane Smith,102,Plants use sunlight...,Water evaporates...,Genes are...
```

**File 3: bob_johnson_response.csv**
```csv
Name,Roll Number,Q1,Q2,Q3
Bob Johnson,103,Cells need light...,Rain falls...,Chromosomes have...
```

---

## How It Works

### Step-by-Step Process

1. **Students Submit Google Forms Individually**
   - Each student fills out their own Google Form
   - Each submission generates a separate CSV file

2. **Teacher Downloads All Responses**
   - Go to Google Forms → Responses tab
   - Click "Download responses (.csv)" for each student
   - OR download all from Google Drive

3. **Upload All Files at Once**
   - Go to Teacher Dashboard: `http://localhost:5000/teacher`
   - Click the upload area or drag & drop
   - **Select ALL student files at once** (Ctrl+Click or Shift+Click)
   - You can upload 10, 20, 50+ files in one go!

4. **System Combines Automatically**
   - Backend merges all files into one dataset
   - Extracts Name, Roll Number, and answers from each file
   - Creates single combined sheet internally

5. **Enter Model Answers**
   - System detects questions from the first file
   - Enter ideal answer for each question

6. **Grade All Students**
   - Click "Grade All Students"
   - System processes combined data
   - Generates individual scoresheets for each student

---

## Google Forms Setup

### Recommended Form Structure

```
Question 1: What is your name?
Answer type: Short answer

Question 2: What is your roll number?
Answer type: Short answer

Question 3: Explain photosynthesis.
Answer type: Paragraph

Question 4: Describe the water cycle.
Answer type: Paragraph

Question 5: What is DNA?
Answer type: Paragraph
```

### Export Individual Responses

**Method 1: Manual Download**
1. Open your Google Form
2. Go to "Responses" tab
3. Click three dots (⋮) next to a response
4. Select "Print" → Save as PDF, OR
5. Use Google Sheets → File → Download → CSV

**Method 2: Google Sheets (Recommended)**
1. Link form to Google Sheets
2. Each row = one student
3. Download sheet as CSV
4. Split into individual files (optional) OR upload the entire sheet

**Method 3: Apps Script (Advanced)**
Create a script to export each row as separate CSV:
```javascript
function exportResponses() {
  var sheet = SpreadsheetApp.getActiveSheet();
  var data = sheet.getDataRange().getValues();
  
  for (var i = 1; i < data.length; i++) {
    var fileName = data[i][0] + "_response.csv"; // Name column
    // Export logic here
  }
}
```

---

## Technical Details

### File Requirements

| Requirement | Details |
|-------------|---------|
| **Format** | CSV (.csv), Excel (.xlsx, .xls) |
| **Size** | Max 16MB per file |
| **Columns** | Must include: `Name`, `Roll Number` |
| **Questions** | Any column names (Q1, Q2, Question 1, etc.) |

### How Combining Works

1. **Parse Each File**
   - Read CSV/Excel format
   - Extract headers and data

2. **Validate Structure**
   - Check for Name and Roll Number columns
   - Ensure question columns exist

3. **Merge Data**
   - Concatenate all rows
   - If a file has multiple rows, take first row only
   - Create single DataFrame with all students

4. **Process as One**
   - Grade combined dataset
   - Generate individual reports

### Backend Logic

```python
def combine_multiple_files(file_paths):
    all_dataframes = []
    
    for file_path in file_paths:
        # Parse each file
        df = pd.read_csv(file_path)  # or pd.read_excel()
        
        # Take first row if multiple rows
        if len(df) > 1:
            df = df.head(1)
        
        all_dataframes.append(df)
    
    # Combine all
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df
```

---

## Usage Examples

### Example 1: 5 Students, 3 Questions

**Upload:**
- `alice_101.csv`
- `bob_102.csv`
- `charlie_103.csv`
- `diana_104.csv`
- `eve_105.csv`

**Each file contains:**
```csv
Name,Roll Number,Q1,Q2,Q3
Alice Brown,101,Answer...,Answer...,Answer...
```

**Result:**
- System combines into 5-row dataset
- Generates 5 individual scoresheets
- Creates 1 class summary

### Example 2: 30 Students, 5 Questions

**Upload:**
Select all 30 CSV files at once

**Processing Time:**
- Upload: 2-3 seconds
- Combining: 1 second
- Grading: 15-20 seconds
- Total: ~20-25 seconds

**Output:**
- 30 individual Excel scoresheets
- 1 class summary Excel file
- All students graded automatically!

---

## Advantages of Multiple Files

### ✅ Benefits

1. **Natural Workflow**
   - Matches how Google Forms exports data
   - No need to manually combine files

2. **Flexible**
   - Upload all at once OR
   - Upload in batches (upload 10 now, 20 later)

3. **Error Recovery**
   - If one file fails, others still process
   - Easy to identify problematic submissions

4. **Privacy**
   - Each student's response stays separate until processing
   - No need to share combined file

### 🔄 Comparison

| Feature | Single File | Multiple Files |
|---------|------------|----------------|
| **Upload** | 1 file | 1-100+ files |
| **Preparation** | Must combine manually | No preparation needed |
| **Google Forms** | Need to merge exports | Use exports directly |
| **Processing** | Fast | Slightly slower (combining step) |
| **Best For** | Pre-combined data | Individual submissions |

---

## Troubleshooting

### Issue: "No valid files uploaded"

**Problem:** Files are not CSV/Excel format

**Solution:**
- Check file extensions (.csv, .xlsx, .xls)
- Convert other formats to CSV first

### Issue: "Missing required columns"

**Problem:** Files don't have Name or Roll Number columns

**Solution:**
- Make sure Google Form has name/roll questions
- Column names must match exactly: "Name", "Roll Number"

### Issue: Different question structures

**Problem:** Files have different column layouts

**Solution:**
- All files must have same question structure
- Use same Google Form for all students
- Don't modify column names

### Issue: Upload takes long time

**Problem:** Too many files or large file sizes

**Solution:**
- Upload in batches (e.g., 20 files at a time)
- Check individual file sizes (max 16MB each)
- Use CSV instead of Excel for faster processing

---

## Best Practices

### 📋 Recommendations

1. **Use Consistent Google Form**
   - Same form for all students
   - Don't change questions mid-collection

2. **File Naming**
   - Use student name in filename
   - Example: `john_doe_101.csv`

3. **Backup**
   - Keep original files before uploading
   - Download results immediately

4. **Batch Processing**
   - If >50 students, consider uploading in groups
   - Prevents timeout issues

5. **Verification**
   - Check first few results
   - Ensure model answers are correct

---

## API Example

### Upload Multiple Files via API

```javascript
const formData = new FormData();

// Add multiple files
for (let file of selectedFiles) {
  formData.append('files[]', file);
}

// Add model answers
formData.append('model_answers', JSON.stringify({
  'Q1': 'Model answer 1...',
  'Q2': 'Model answer 2...',
  'Q3': 'Model answer 3...'
}));

// Send request
const response = await fetch('/batch-grade', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(`Graded ${result.total_students} students from ${selectedFiles.length} files`);
```

---

## Future Enhancements

### Planned Features

- [ ] Progress indicator for each file during combining
- [ ] Validation report before grading (show detected students)
- [ ] Support for zip file upload (extract automatically)
- [ ] Email individual scoresheets to students
- [ ] Merge partial submissions (student submits multiple times)

---

## Summary

**Multiple file upload** is perfect for:
- ✅ Individual Google Form submissions
- ✅ One file per student workflow
- ✅ Avoiding manual data combination
- ✅ Scalable to 100+ students

**Single file upload** is perfect for:
- ✅ Pre-combined datasets
- ✅ Quick testing
- ✅ Google Sheets exports (all students in one sheet)
- ✅ Faster processing

**Both modes work seamlessly!** Choose what fits your workflow best.

---

**Updated:** November 3, 2025  
**Version:** 2.0 - Multiple Files Support ✨
