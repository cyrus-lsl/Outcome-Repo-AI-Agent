# Measurement Instrument Assistant

An AI-powered web application for finding and recommending measurement instruments for research purposes. Built with Streamlit and powered by Hugging Face AI models with semantic search capabilities.

## Features

- 🤖 **AI-powered chat interface** - Natural language queries using Hugging Face models
- 🔍 **Semantic search** - Advanced embedding-based search for better relevance
- 📊 **Excel data integration** - Seamless integration with measurement instrument databases
- 💬 **Intelligent recommendations** - Context-aware instrument suggestions
- 🎯 **Manual search filters** - Advanced filtering by beneficiaries, measures, validation status
- ⚡ **Performance optimized** - Caching and efficient data loading
- 🔒 **Production ready** - Comprehensive error handling, logging, and security

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Hugging Face API token ([Get one here](https://huggingface.co/settings/tokens))
- Excel file with measurement instruments data

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Outcome Repo Agent"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp env.example .env
   # Edit .env and add your HF_TOKEN
   ```

4. **Run the application**
   ```bash
   streamlit run frontend/app.py
   ```

   Or using the CLI:
   ```bash
   python run_agent_cli.py
   ```

## Configuration

The application can be configured via environment variables. See `env.example` for all available options.

### Required Configuration

- `HF_TOKEN` - Your Hugging Face API token (required)

### Optional Configuration

- `EXCEL_FILE_PATH` - Path to your Excel file (default: `measurement_instruments.xlsx`)
- `EXCEL_SHEET_NAME` - Sheet name in Excel file (default: `Measurement Instruments`)
- `MAX_RESULTS` - Maximum number of results to return (default: `8`)
- `SEMANTIC_SEARCH_ENABLED` - Enable semantic search (default: `true`)
- `ENABLE_CACHING` - Enable response caching (default: `true`)

## Project Structure

```
Outcome Repo Agent/
├── frontend/
│   └── app.py              # Streamlit web interface
├── backend/
│   ├── agent_core.py       # Core agent logic with semantic search
│   └── outcome_repo_agent.py
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
├── Procfile                # Heroku deployment config
├── run_agent_cli.py        # CLI runner
├── env.example             # Environment variable template
└── README.md               # This file
```

## Usage

### Web Interface

1. Start the application: `streamlit run frontend/app.py`
2. Navigate to the Chat page for AI-powered search
3. Use Manual Search for advanced filtering options

### Chat Interface

- Enter natural language queries (e.g., "mental health assessment for elderly")
- Use checkboxes to filter by HK validation or programme-level metrics
- Results include detailed instrument information with badges and expandable sections

### Manual Search

- Filter by target beneficiaries (comma-separated)
- Filter by measurement type
- Apply additional filters for validation status and programme-level metrics

## Deployment

### Docker

```bash
docker build -t measurement-instrument-assistant .
docker run -p 8501:8501 -e HF_TOKEN=your_token measurement-instrument-assistant
```

### Heroku

1. Set environment variables in Heroku dashboard
2. Deploy using the Procfile

## Security & Best Practices

- ✅ Input sanitization to prevent injection attacks
- ✅ Environment variable validation
- ✅ Comprehensive error handling and logging
- ✅ Rate limiting considerations (implement at infrastructure level)
- ✅ Secure credential management (never commit `.env` files)

## Logging

The application logs to both:
- Console output (for development)
- `app.log` file (for production monitoring)

Log levels can be configured in `config.py`.

## Troubleshooting

### Common Issues

1. **"HF_TOKEN not set" error**
   - Ensure `.env` file exists with `HF_TOKEN` set
   - Check that token is valid and has API access

2. **Excel file not found**
   - Verify `EXCEL_FILE_PATH` points to correct file
   - Check file permissions

3. **Semantic search not working**
   - Ensure `sentence-transformers` is installed
   - Check `SEMANTIC_SEARCH_ENABLED` setting

## Contributing

1. Follow Python PEP 8 style guidelines
2. Add logging for new features
3. Update documentation for API changes
4. Test thoroughly before submitting

## License

[Add your license here]

## Support

For issues or questions, please contact the development team or open an issue in the repository.