# Multi-Agent GitHub Issue Routing System

An AI-powered system that automatically routes GitHub issues to a swarm of 50+ specialized AI agents who collaboratively analyze, discuss, and provide expert recommendations through a moderated multi-round deliberation process.

[![GitHub Issues](https://img.shields.io/github/issues/vtggit/CodeAgent)](https://github.com/vtggit/CodeAgent/issues)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

When a GitHub issue is created, this system:
1. **Automatically identifies** relevant expert agents (UI Architect, Security Expert, ADA Specialist, etc.)
2. **Agents analyze** the issue in parallel (Round 1)
3. **Agents collaborate** by responding to each other's insights (Rounds 2-N)
4. **Moderator detects** convergence and concludes discussion
5. **System posts** comprehensive recommendations back to the GitHub issue

## Key Innovation

Unlike traditional workflow engines (sequential/DAG-based), this implements a **"Moderated Multi-Agent Discourse"** execution model where:
- Agents dynamically determine when to participate
- Agents can respond to each other's insights
- Conversation naturally converges without endless rambling
- AI-powered convergence detection prevents infinite loops

## Architecture

```
GitHub Issue → Webhook → Redis Queue → Worker
                                         ↓
                            ╔═══════════════════════╗
                            ║  Deliberation Loop    ║
                            ║  ┌─────────────────┐  ║
                            ║  │ Moderator       │  ║
                            ║  │ selects agents  │  ║
                            ║  └────────┬────────┘  ║
                            ║           │           ║
                            ║  ┌────────▼────────┐  ║
                            ║  │ Round Executor  │  ║
                            ║  │ (parallel)      │  ║
                            ║  └────────┬────────┘  ║
                            ║           │           ║
                            ║  ┌────────▼────────┐  ║
                            ║  │ Convergence     │  ║
                            ║  │ Detection       │  ║
                            ║  └────────┬────────┘  ║
                            ║           │           ║
                            ║    Continue? No → Exit║
                            ║           │ Yes       ║
                            ║           └──────┐    ║
                            ║                  │    ║
                            ╚══════════════════╪════╝
                                               │
                            GitHub Issue ← Summary
```

## Features

- 🤖 **50+ Specialized Agents**: Architecture, Development, QA, Security, Accessibility, Performance, etc.
- 🔄 **Round-Based Deliberation**: Multi-round discussions with natural convergence
- 🧠 **AI-Powered Moderation**: Intelligent agent selection and convergence detection
- 🔌 **Multi-LLM Support**: Claude, GPT-4, local LM Studio models via LiteLLM
- 📊 **Real-time Monitoring**: Prometheus metrics + Grafana dashboards
- 🔒 **Production-Ready**: Webhook validation, secrets management, error tracking
- 🐳 **Docker Support**: Complete containerization with Docker Compose

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn
- **AI Layer**: Claude Agent SDK, LiteLLM
- **Data**: SQLite, Redis, SQLAlchemy
- **GitHub**: PyGithub, Webhook handlers
- **Monitoring**: Prometheus, Grafana, Sentry, Structlog
- **Testing**: pytest, pytest-asyncio
- **Deployment**: Docker, Docker Compose, GitHub Actions

## Prerequisites

- Python 3.11 or higher
- Redis (for message queue)
- GitHub account and personal access token
- Anthropic API key (for Claude)
- Optional: OpenAI API key, LM Studio for local models

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/vtggit/CodeAgent.git
cd CodeAgent
./init.sh
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your credentials:

```bash
# Required
GITHUB_TOKEN=ghp_your_token_here
ANTHROPIC_API_KEY=sk-ant-your_key_here

# Optional
DATABASE_URL=sqlite:///./data/multi_agent.db
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
MAX_ROUNDS=10
CONVERGENCE_THRESHOLD=0.8
```

### 3. Run with Docker Compose (Recommended)

```bash
docker-compose up
```

### 4. Or Run Manually

**Terminal 1 - API Server:**
```bash
source venv/bin/activate
uvicorn src.api.main:app --reload
```

**Terminal 2 - Worker:**
```bash
source venv/bin/activate
python src.orchestration/worker.py
```

**Terminal 3 - Redis:**
```bash
redis-server
```

### 5. Configure GitHub Webhook

1. Go to your repository → Settings → Webhooks → Add webhook
2. Payload URL: `https://your-domain.com/webhook/github`
3. Content type: `application/json`
4. Secret: Generate a random string and add to `.env` as `WEBHOOK_SECRET`
5. Events: Select "Issues" and "Issue comments"

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Project Structure

```
multi-agent-github-system/
├── src/
│   ├── api/              # FastAPI webhook endpoints
│   ├── orchestration/    # Main deliberation engine
│   ├── agents/           # Agent implementations
│   ├── models/           # Pydantic data models
│   ├── integrations/     # GitHub, Redis, Claude SDK
│   ├── visualization/    # Mermaid diagram generation
│   └── utils/            # Config, logging, metrics
├── config/
│   ├── agent_definitions.yaml    # 50+ agent configs
│   └── workflow_templates.yaml   # Workflow templates
├── tests/                # Unit and integration tests
├── scripts/              # Deployment and utility scripts
├── docs/                 # Additional documentation
└── .github/
    └── workflows/        # CI/CD pipelines
```

## Agent Catalog

The system includes 50+ specialized agents organized by domain:

### Architecture & Design
- System Architect, UI Architect, Data Architect, API Architect

### Development
- Frontend Dev, Backend Dev, iOS Developer, Android Developer, Database Expert

### Quality & Compliance
- QA Engineer, Security Expert, ADA/Accessibility Expert, Performance Expert, Privacy Expert

### Infrastructure & DevOps
- DevOps Engineer, Cloud Architect, SRE, Network Engineer

### Data & ML
- ML Engineer, Data Engineer, Analytics Expert, Data Scientist

### Business & Product
- Product Manager, UX Researcher, UX Designer, UI Designer, Tech Writer

### Specialized Domains
- SEO Expert, i18n Expert, Payment Systems, Email Systems, Search Expert, and more...

See [docs/AGENTS.md](docs/AGENTS.md) for complete catalog.

## Development

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit

# With coverage
pytest --cov=src --cov-report=html

# Integration tests (requires setup)
pytest tests/integration -m integration
```

### Code Quality

```bash
# Linting
ruff check src/

# Type checking
mypy src/

# Format code
black src/ tests/
```

## Monitoring

### Prometheus Metrics

Access at http://localhost:9090

Key metrics:
- `deliberations_total` - Total number of deliberations
- `deliberation_rounds` - Histogram of rounds per deliberation
- `agent_participation_rate` - Agent participation rates
- `convergence_time_seconds` - Time to convergence

### Grafana Dashboards

Access at http://localhost:3000 (default credentials: admin/admin)

Import dashboard from `monitoring/grafana-dashboard.json`

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed deployment guides:
- Docker deployment
- Kubernetes with Helm
- AWS/GCP/Azure cloud platforms
- Security best practices

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details

## Support

- **Issues**: https://github.com/vtggit/CodeAgent/issues
- **Discussions**: https://github.com/vtggit/CodeAgent/discussions
- **Documentation**: https://github.com/vtggit/CodeAgent/wiki

## Acknowledgments

Built with:
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) by Anthropic
- [FastAPI](https://fastapi.tiangolo.com/) web framework
- [LiteLLM](https://github.com/BerriAI/litellm) for multi-LLM support

---

**Status**: Active Development | **Version**: 0.1.0 | **Python**: 3.11+
