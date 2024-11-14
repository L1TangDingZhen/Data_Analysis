# Data Type Analyzer with AI-Enhanced Type Inference

This project is a web application for analyzing and converting data types in datasets, with a focus on intelligent type inference and complex data handling.

## Key Features

### 1. Pandas Data Type Inference and Conversion
- Advanced data type inference using rule-based methods and an AI model (spaCy)
- Smart handling of mixed data types and non-standard formats
- Support for complex data patterns through AI analysis
- Optimization for large datasets with chunk processing
- Handling of common data type issues:
  - Columns defaulting to 'object' type
  - Mixed data types in single columns
  - Non-standard date formats
  - Special values and null representations

### 2. Supported Data Types
- Text (`object`)
- Numbers (`int8`, `int16`, `int32`, `int64`, `float32`, `float64`)
- Dates (`datetime64[ns]`)
- Boolean
- Category
- Complex mixed types

### 3. Special Value Handling
- Intelligent handling of:
  - NULL values
  - Missing data
  - Currency formats
  - Percentage values
  - Mixed date formats
  - Special characters

### 4. Performance Optimization
- Memory usage optimization through data type downcasting
- Chunk processing for large files
- Caching mechanisms for processed data
- Efficient data type conversion strategies

### 5. Error Handling
- Comprehensive error detection and reporting
- Graceful fallback mechanisms
- Detailed logging system
- User-friendly error messages

## Technical Architecture

### Backend (Django REST Framework)
- RESTful API endpoints
- File processing system
- Data type inference engine
- AI model integration (spaCy)
- Statistical analysis

### Frontend (React)
- Intuitive file upload interface
- Real-time data preview
- Type modification capabilities
- Statistical visualization
- Export functionality

## Getting Started

### Prerequisites
- Docker
- Docker Compose

### Installation and Running

```bash
# Clone the repository
git clone https://github.com/L1TangDingZhen/Data_Analysis


# Start the application using Docker Compose
docker-compose up --build


## Accessing the Application

- **Frontend**: [http://localhost:3000](http://localhost:3000)
- **Backend API**: [http://localhost:8000](http://localhost:8000)

## Usage

### Upload Data
- Support for CSV and Excel files
- Automatic type inference
- Initial data preview

### Analyze Types
- View inferred data types
- Check sample values
- See type inference confidence scores

### Modify Types
- Change data types manually
- Preview changes instantly
- Revert changes if needed

### Export and Statistics
- Export processed data
- View detailed statistics
- Analyze data distributions

## Technical Implementation Details

### AI Model Integration
- Uses spaCy's `en_core_web_md` model
- Enhances type inference accuracy
- Handles complex data patterns
- Provides confidence scores

### Data Processing Pipeline
- Initial file reading
- Special value cleaning
- AI-enhanced type inference
- Type conversion
- Performance optimization

### Performance Features
- Efficient memory usage
- Optimized data types
- Chunk processing for large files
- Caching mechanisms

## Testing

The project includes comprehensive test coverage:
- Unit tests for core functionality
- Integration tests for API endpoints
- Complex data handling tests
- Performance tests
- Error handling tests

## Project Requirements Fulfilled

- ✅ Pandas data type inference and conversion
- ✅ Handling of default 'object' type columns
- ✅ Complex data type processing
- ✅ Large file handling
- ✅ Performance optimization
- ✅ Comprehensive error handling
- ✅ Clean and maintainable code
- ✅ User-friendly interface
- ✅ Detailed documentation
