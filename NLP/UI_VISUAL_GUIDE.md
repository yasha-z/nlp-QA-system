# 📸 Visual Guide - Multiple File Upload

## New UI Features

### 1. Updated Upload Area

**Before:**
```
┌─────────────────────────────────────┐
│  📤 Upload Student Responses         │
├─────────────────────────────────────┤
│  Drop your file here or click       │
│  Supports CSV, Excel - Max 16MB     │
└─────────────────────────────────────┘
```

**After:**
```
┌─────────────────────────────────────┐
│  📤 Upload Student Responses         │
├─────────────────────────────────────┤
│  Drop your file(s) here or click    │
│  Supports CSV, Excel - Max 16MB     │
│  ✨ NEW: Select multiple files!     │
└─────────────────────────────────────┘
```

---

### 2. File List Display

When you upload multiple files, you'll see:

```
┌─────────────────────────────────────┐
│  Uploaded Files:                     │
├─────────────────────────────────────┤
│  📄 john_doe_101.csv                │
│  📄 jane_smith_102.csv              │
│  📄 bob_johnson_103.csv             │
│  📄 alice_brown_104.csv             │
│  📄 charlie_davis_105.csv           │
├─────────────────────────────────────┤
│                    [Clear All] ←─── │
└─────────────────────────────────────┘
```

---

### 3. Updated Instructions

The instructions section now shows both methods:

```
📋 How to Use

1. Prepare your files:
   
   Option 1: Upload ONE combined file
   ├─ CSV/Excel with all students
   └─ Name, Roll Number, Q1, Q2 columns
   
   Option 2: Upload MULTIPLE files ✨
   ├─ One Google Form per student
   └─ They will be combined automatically!

2. Upload file(s): Drag & drop or browse

3. Set model answers: Enter ideal answers

4. Grade: Click "Grade All Students"

5. Download: Individual scoresheets + summary
```

---

### 4. Processing Progress

When grading multiple files:

```
┌─────────────────────────────────────┐
│  ⚡ Processing...                    │
├─────────────────────────────────────┤
│  ✓ Combined 30 student files        │
│  ⏳ Grading answers...              │
│  ⏳ Generating scoresheets...       │
│                                     │
│  ████████████░░░░░░░░░ 60%         │
└─────────────────────────────────────┘
```

---

### 5. Results Display

After successful processing:

```
┌─────────────────────────────────────┐
│  ✅ Grading Complete!                │
├─────────────────────────────────────┤
│  Successfully graded 30 students    │
│  from 30 file(s)                    │
│                                     │
│  📊 Class Statistics                │
│  ├─ Average: 75.5%                 │
│  ├─ Highest: 95.0%                 │
│  └─ Lowest: 45.0%                  │
│                                     │
│  📥 Download Results                │
│  [Class Summary]  [All Scoresheets]│
└─────────────────────────────────────┘
```

---

## File Selection Dialog

### Windows Explorer

When you click "Browse" or drag files:

```
┌───────────────────────────────────────────┐
│  Select files to upload                   │
├───────────────────────────────────────────┤
│  Name                          Size       │
│  ☑ john_doe_101.csv           2 KB       │
│  ☑ jane_smith_102.csv         2 KB       │
│  ☑ bob_johnson_103.csv        2 KB       │
│  ☑ alice_brown_104.csv        2 KB       │
│  ☑ charlie_davis_105.csv      2 KB       │
│                                           │
│  5 files selected                         │
│                                           │
│            [Cancel]  [Open] ←─────────── │
└───────────────────────────────────────────┘
```

**Tips:**
- **Ctrl+Click**: Select individual files
- **Shift+Click**: Select range of files
- **Ctrl+A**: Select all files in folder

---

## Drag & Drop Animation

### Step 1: Hover
```
┌─────────────────────────────────────┐
│  📤 DRAG FILES HERE                  │
│                                     │
│  [Dashed border pulsing]            │
│                                     │
│  Release to upload!                 │
└─────────────────────────────────────┘
```

### Step 2: Drop
```
┌─────────────────────────────────────┐
│  ⏳ Uploading...                     │
│                                     │
│  ████████████████████░░ 85%        │
│                                     │
│  5 of 5 files uploaded              │
└─────────────────────────────────────┘
```

### Step 3: Success
```
┌─────────────────────────────────────┐
│  ✅ Files uploaded successfully!     │
│                                     │
│  📄 5 files ready for processing    │
│                                     │
│  Next: Enter model answers below    │
└─────────────────────────────────────┘
```

---

## Model Answers Section

Looks the same, but shows combined question count:

```
┌─────────────────────────────────────┐
│  📝 Enter Model Answers              │
├─────────────────────────────────────┤
│  Detected 3 questions from files    │
│  (Combined from 30 student files)   │
│                                     │
│  Q1: ________________________________│
│  Photosynthesis is the process...   │
│  ____________________________________│
│                                     │
│  Q2: ________________________________│
│  The water cycle involves...        │
│  ____________________________________│
│                                     │
│  Q3: ________________________________│
│  DNA contains genetic...            │
│  ____________________________________│
│                                     │
│         [Grade All Students]        │
└─────────────────────────────────────┘
```

---

## Mobile View

### Responsive Design

**On Phone/Tablet:**

```
┌─────────────────┐
│  📤 Upload      │
├─────────────────┤
│  Tap to select  │
│  Multiple files │
│  supported!     │
│                 │
│  [Browse]       │
└─────────────────┘

┌─────────────────┐
│  Files (5):     │
├─────────────────┤
│  📄 student1... │
│  📄 student2... │
│  📄 student3... │
│  📄 student4... │
│  📄 student5... │
│  [Clear]        │
└─────────────────┘
```

---

## Color Scheme

### File Upload Area

- **Default**: Light blue background (#f0f9ff)
- **Hover**: Blue border (#3b82f6)
- **Drag Over**: Green highlight (#22c55e)
- **Error**: Red border (#ef4444)

### File List Items

- **Background**: White (#ffffff)
- **Border**: Light gray (#e2e8f0)
- **Text**: Dark gray (#334155)
- **Icon**: Blue (#3b82f6)

### Buttons

- **Clear All**: Red (#ef4444)
- **Grade**: Green (#22c55e)
- **Download**: Blue (#3b82f6)

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Ctrl+O** | Open file dialog |
| **Ctrl+A** | Select all files (in dialog) |
| **Delete** | Clear uploaded files |
| **Enter** | Start grading (when ready) |
| **Esc** | Cancel upload |

---

## Status Indicators

### Upload States

1. **Ready** (🔵)
   - Upload area visible
   - No files selected

2. **Uploading** (🟡)
   - Progress bar showing
   - Files being transferred

3. **Uploaded** (🟢)
   - Files list displayed
   - Ready for model answers

4. **Processing** (🟠)
   - Combining files
   - Grading in progress

5. **Complete** (✅)
   - Results ready
   - Download available

6. **Error** (🔴)
   - Something went wrong
   - Clear error message

---

## Example Workflows

### Workflow 1: Quick Single File

```
User Action          System Response
────────────────────────────────────────
Drop 1 file      →  ✓ File uploaded
Enter answers    →  ✓ Ready to grade
Click Grade      →  ⏳ Processing...
Wait 5s          →  ✅ Done! Download
```

### Workflow 2: Multiple Files

```
User Action               System Response
───────────────────────────────────────────
Select 30 files       →  ⏳ Uploading...
Wait 3s               →  ✓ 30 files ready
System combines       →  ✓ Combined dataset
Enter answers         →  ✓ Ready to grade
Click Grade           →  ⏳ Processing 30...
Wait 20s              →  ✅ Done! 30 results
Download summary      →  📊 Class report
Download individuals  →  📄 30 scoresheets
```

### Workflow 3: Error Handling

```
User Action          System Response
────────────────────────────────────────
Drop .txt file   →  ❌ Invalid format
Drop .csv file   →  ✓ File accepted
Large 20MB file  →  ❌ Size too large
Valid file       →  ✓ Proceed to grade
```

---

## Browser Compatibility

| Browser | Multiple Upload | Drag & Drop | Progress |
|---------|----------------|-------------|----------|
| Chrome 90+ | ✅ | ✅ | ✅ |
| Firefox 88+ | ✅ | ✅ | ✅ |
| Edge 90+ | ✅ | ✅ | ✅ |
| Safari 14+ | ✅ | ✅ | ✅ |
| Opera 76+ | ✅ | ✅ | ✅ |

---

## Accessibility

### Screen Reader Support

- File upload: "Upload area, drop files or click to browse"
- File count: "5 files selected for upload"
- Progress: "Processing, 60 percent complete"
- Results: "Grading complete, 30 students graded successfully"

### Keyboard Navigation

- Tab through all interactive elements
- Enter to activate buttons
- Space to toggle selections
- Arrow keys for file list navigation

---

## Sample Files Provided

The dashboard includes a "Download Sample" button:

```
┌─────────────────────────────────────┐
│  Need help? Download sample files   │
├─────────────────────────────────────┤
│  [Sample Single File (All Students)]│
│  [Sample Multiple Files (Zip)]      │
│  [Template (Empty)]                 │
└─────────────────────────────────────┘
```

Contents:
- `sample_combined.csv` - 5 students in one file
- `sample_individual_*.csv` - 5 separate files
- `template.csv` - Empty structure to fill in

---

## Animation Details

### Loading Spinner

```
⏳ Processing...

   ◜
  ◝ ◟
   ◞

(Rotates continuously)
```

### Progress Bar

```
████████████░░░░░░░░░ 60%

Colors:
- Complete: Green (#22c55e)
- Remaining: Gray (#e2e8f0)
- Border: Dark gray (#334155)
```

### Success Checkmark

```
✅

Animates:
1. Fade in
2. Scale up (0.8 → 1.2 → 1.0)
3. Bounce effect
```

---

## Summary

### UI Improvements

✅ **Visual Feedback**
- File list shows all uploads
- Progress indicators
- Status messages

✅ **User-Friendly**
- Drag & drop anywhere
- Clear instructions
- Error messages

✅ **Responsive**
- Works on desktop/mobile
- Adapts to screen size
- Touch-friendly

✅ **Accessible**
- Screen reader support
- Keyboard navigation
- High contrast

---

**Guide Version:** 1.0  
**Last Updated:** November 3, 2025  
**Access Dashboard:** http://localhost:5000/teacher
