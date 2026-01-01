-- Database Initialization Script
-- Run this on PostgreSQL startup to initialize the database

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search
CREATE EXTENSION IF NOT EXISTS "btree_gin";  -- For JSONB indexing

-- Tables will be created automatically by SQLAlchemy
-- This script is for any manual setup needed
