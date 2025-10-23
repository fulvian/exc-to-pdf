# Production Deployment Guide

This document outlines the production deployment setup for exc-to-pdf.

## Overview

The exc-to-pdf project is configured for automated deployment with:
- **CI/CD Pipeline**: GitHub Actions for testing and publishing
- **Documentation**: MkDocs deployed to GitHub Pages
- **Package Distribution**: PyPI publishing via GitHub Actions
- **Version Management**: Semantic versioning with automated releases

## Deployment Architecture

```
GitHub Repository → GitHub Actions → PyPI
                → GitHub Actions → GitHub Pages (Documentation)
```

## Pre-deployment Checklist

### 1. Repository Setup
- [x] Repository initialized with git
- [x] Remote configured (GitHub)
- [x] Main branch set to `main`
- [x] GitHub Actions workflows configured

### 2. Package Configuration
- [x] `pyproject.toml` configured with production metadata
- [x] `MANIFEST.in` includes all necessary files
- [x] `LICENSE` file created (MIT)
- [x] `CHANGELOG.md` initialized
- [x] Package builds successfully

### 3. CI/CD Configuration
- [x] CI workflow (`.github/workflows/ci.yml`)
- [x] Release workflow (`.github/workflows/release.yml`)
- [x] Documentation workflow (`.github/workflows/docs.yml`)
- [x] All workflows tested

### 4. Documentation
- [x] MkDocs configuration updated for GitHub Pages
- [x] Documentation builds successfully
- [x] Navigation configured
- [x] Theme and styling applied

## Deployment Process

### Automated Release (Recommended)

1. **Make changes** to the codebase
2. **Update version** in `pyproject.toml` (semantic versioning)
3. **Update CHANGELOG.md** with release notes
4. **Commit and push** changes to main branch
5. **Create release tag**:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
6. **GitHub Actions** will automatically:
   - Run tests on all Python versions
   - Build package distributions
   - Publish to PyPI
   - Create GitHub Release
   - Deploy documentation

### Manual Release (Alternative)

Use the provided release script:
```bash
python scripts/release.py v1.0.0
```

This will:
- Run tests with coverage
- Build package
- Commit changes
- Create and push tag
- Trigger automated deployment

## Monitoring

### GitHub Actions
- Monitor workflow runs in GitHub repository Actions tab
- Check for test failures
- Verify successful PyPI publishing
- Confirm documentation deployment

### PyPI
- Monitor package at https://pypi.org/project/exc-to-pdf/
- Verify download statistics
- Check package metadata

### GitHub Pages
- Documentation available at https://fulvian.github.io/exc-to-pdf/
- Verify all pages load correctly
- Check navigation and links

## Quality Gates

### Testing Requirements
- ✅ All tests must pass on Python 3.9-3.12
- ✅ Coverage must be ≥ 90%
- ✅ Linting must pass (flake8, black, mypy)
- ✅ Package must build without warnings

### Package Quality
- ✅ Valid pyproject.toml configuration
- ✅ All dependencies properly specified
- ✅ Entry points configured correctly
- ✅ Documentation included in package

### Documentation Requirements
- ✅ MkDocs builds successfully
- ✅ All navigation links work
- ✅ No broken internal links
- ✅ Responsive design verified

## Security Considerations

### PyPI Publishing
- Uses OIDC authentication (no API keys required)
- Trusted publishing configured in PyPI
- Automated security scanning via GitHub Actions

### Repository Security
- No secrets committed to repository
- GitHub Actions permissions properly configured
- Dependabot configured for dependency updates

## Troubleshooting

### Common Issues

**Build Fails**
- Check `pyproject.toml` syntax
- Verify all dependencies available
- Check MANIFEST.in for missing files

**Tests Fail**
- Update test cases for new functionality
- Check Python version compatibility
- Verify dependency versions

**Documentation Deploy Fails**
- Check mkdocs.yml configuration
- Verify all referenced files exist
- Check for broken links

**PyPI Publish Fails**
- Verify version number is incremented
- Check PyPI project configuration
- Ensure OIDC trust relationship

## Maintenance

### Regular Tasks
- Update dependencies in `pyproject.toml`
- Keep CHANGELOG.md current
- Monitor test coverage
- Update documentation

### Release Process
1. Update version in `pyproject.toml`
2. Add changelog entry
3. Commit changes
4. Create and push tag
5. Verify automated deployment

## Success Metrics

- ✅ Package successfully published to PyPI
- ✅ Documentation deployed to GitHub Pages
- ✅ All tests passing in CI/CD
- ✅ Zero security vulnerabilities
- ✅ 95%+ test coverage maintained
- ✅ Documentation builds without warnings

## Support

For deployment issues:
1. Check GitHub Actions logs
2. Review package build output
3. Verify configuration files
4. Consult this deployment guide

---

**Deployment Status**: Ready for production use
**Last Updated**: 2025-10-22
**Next Release**: v1.0.0