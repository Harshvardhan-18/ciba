"""initial_schema

Revision ID: 19baeac8e423
Revises: 
Create Date: 2026-08-03 23:01:35.348105

NOTE: The schema has a intentional circular FK between campaigns
(selected_concept_id → creative_concepts) and creative_concepts
(campaign_id → campaigns). We break the cycle by:
  1. Creating campaigns WITHOUT the selected_concept_id FK.
  2. Creating creative_concepts with its campaign_id FK.
  3. Adding the selected_concept_id FK to campaigns via add_column / create_foreign_key.

Similarly, creative_assets.approved_attempt_id → generation_attempts creates
a second cycle (creative_assets → generation_attempts → creative_assets via asset_id).
We break this by adding approved_attempt_id as a nullable column after both tables exist.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '19baeac8e423'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -----------------------------------------------------------------------
    # Leaf tables with no inbound FKs from the problem cycles (create first)
    # -----------------------------------------------------------------------

    op.create_table(
        'users',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('google_sub', sa.String(length=255), nullable=False),
        sa.Column('email', sa.String(length=320), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('avatar_url', sa.String(length=1000), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_google_sub'), 'users', ['google_sub'], unique=True)

    op.create_table(
        'brands',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('owner_id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('logo_url', sa.String(length=1000), nullable=True),
        sa.Column('primary_colors', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('secondary_colors', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('fonts', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('tone', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_brands_owner_id'), 'brands', ['owner_id'], unique=False)

    op.create_table(
        'products',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('brand_id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('product_images', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('extra_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['brand_id'], ['brands.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_products_brand_id'), 'products', ['brand_id'], unique=False)

    # -----------------------------------------------------------------------
    # campaigns — created WITHOUT selected_concept_id FK to break the cycle
    # -----------------------------------------------------------------------

    op.create_table(
        'campaigns',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('brand_id', sa.UUID(), nullable=False),
        sa.Column('product_id', sa.UUID(), nullable=False),
        sa.Column('owner_id', sa.UUID(), nullable=False),
        sa.Column('brief_text', sa.Text(), nullable=False),
        sa.Column('target_audience', sa.String(length=500), nullable=True),
        sa.Column(
            'status',
            sa.Enum('DRAFT', 'GENERATING_CONCEPTS', 'CONCEPTS_READY', 'GENERATING_ASSETS',
                    'COMPLETE', 'FAILED', name='campaign_status'),
            nullable=False,
        ),
        # selected_concept_id added below after creative_concepts exists
        sa.Column('selected_concept_id', sa.UUID(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['brand_id'], ['brands.id'], ),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['product_id'], ['products.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_campaigns_brand_id'), 'campaigns', ['brand_id'], unique=False)
    op.create_index(op.f('ix_campaigns_owner_id'), 'campaigns', ['owner_id'], unique=False)
    op.create_index(op.f('ix_campaigns_product_id'), 'campaigns', ['product_id'], unique=False)

    # -----------------------------------------------------------------------
    # creative_concepts — references campaigns (now exists)
    # -----------------------------------------------------------------------

    op.create_table(
        'creative_concepts',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('campaign_id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('visual_dna', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('copy', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('rationale', sa.Text(), nullable=False),
        sa.Column(
            'status',
            sa.Enum('PROPOSED', 'SELECTED', 'REJECTED', name='concept_status'),
            nullable=False,
        ),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['campaign_id'], ['campaigns.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_creative_concepts_campaign_id'), 'creative_concepts', ['campaign_id'], unique=False)

    # Now add the FK from campaigns.selected_concept_id → creative_concepts.id
    op.create_foreign_key(
        'fk_campaigns_selected_concept_id',
        'campaigns', 'creative_concepts',
        ['selected_concept_id'], ['id'],
    )

    # -----------------------------------------------------------------------
    # creative_assets — created WITHOUT approved_attempt_id FK to break cycle
    # -----------------------------------------------------------------------

    op.create_table(
        'creative_assets',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('campaign_id', sa.UUID(), nullable=False),
        sa.Column('concept_id', sa.UUID(), nullable=False),
        sa.Column(
            'platform',
            sa.Enum('INSTAGRAM', 'WEBSITE', name='platform'),
            nullable=False,
        ),
        sa.Column(
            'placement',
            sa.Enum('IG_FEED', 'IG_STORY', 'WEBSITE_HERO', name='placement'),
            nullable=False,
        ),
        sa.Column('aspect_ratio', sa.String(length=10), nullable=False),
        sa.Column('width', sa.Integer(), nullable=False),
        sa.Column('height', sa.Integer(), nullable=False),
        sa.Column('asset_spec', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            'status',
            sa.Enum('PENDING', 'GENERATING', 'EVALUATING', 'APPROVED', 'MANUAL_REVIEW',
                    'INFRA_FAILED', name='asset_status'),
            nullable=False,
        ),
        # approved_attempt_id FK added below after generation_attempts exists
        sa.Column('approved_attempt_id', sa.UUID(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['campaign_id'], ['campaigns.id'], ),
        sa.ForeignKeyConstraint(['concept_id'], ['creative_concepts.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_creative_assets_campaign_id'), 'creative_assets', ['campaign_id'], unique=False)
    op.create_index(op.f('ix_creative_assets_concept_id'), 'creative_assets', ['concept_id'], unique=False)

    # -----------------------------------------------------------------------
    # generation_attempts — references creative_assets (now exists)
    # -----------------------------------------------------------------------

    op.create_table(
        'generation_attempts',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('asset_id', sa.UUID(), nullable=False),
        sa.Column('attempt_number', sa.Integer(), nullable=False),
        sa.Column('prompt_used', sa.Text(), nullable=False),
        sa.Column('corrective_instruction', sa.Text(), nullable=True),
        sa.Column('image_url', sa.String(length=1000), nullable=True),
        sa.Column('provider', sa.String(length=50), nullable=False),
        sa.Column('infra_failed', sa.Boolean(), nullable=False),
        sa.Column('infra_error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['asset_id'], ['creative_assets.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_generation_attempts_asset_id'), 'generation_attempts', ['asset_id'], unique=False)

    # Now add the FK from creative_assets.approved_attempt_id → generation_attempts.id
    op.create_foreign_key(
        'fk_creative_assets_approved_attempt_id',
        'creative_assets', 'generation_attempts',
        ['approved_attempt_id'], ['id'],
    )

    # -----------------------------------------------------------------------
    # evaluations — references generation_attempts (now exists)
    # -----------------------------------------------------------------------

    op.create_table(
        'evaluations',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('attempt_id', sa.UUID(), nullable=False),
        sa.Column('vlm_product_score', sa.Float(), nullable=False),
        sa.Column('siglip_similarity', sa.Float(), nullable=False),
        sa.Column('ocr_text_score', sa.Float(), nullable=False),
        sa.Column('product_fidelity', sa.Float(), nullable=False),
        sa.Column('brand_consistency', sa.Float(), nullable=False),
        sa.Column('composition_score', sa.Float(), nullable=False),
        sa.Column('prompt_alignment', sa.Float(), nullable=False),
        sa.Column('overall_score', sa.Float(), nullable=False),
        sa.Column('critical_text_error', sa.Boolean(), nullable=False),
        sa.Column('passed', sa.Boolean(), nullable=False),
        sa.Column('failure_reason', sa.Text(), nullable=True),
        sa.Column('vlm_provider', sa.String(length=50), nullable=False),
        sa.Column('raw_response', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['attempt_id'], ['generation_attempts.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_evaluations_attempt_id'), 'evaluations', ['attempt_id'], unique=True)


def downgrade() -> None:
    # Drop in reverse order of creation, FK constraints first where needed.

    op.drop_index(op.f('ix_evaluations_attempt_id'), table_name='evaluations')
    op.drop_table('evaluations')

    op.drop_index(op.f('ix_generation_attempts_asset_id'), table_name='generation_attempts')
    op.drop_table('generation_attempts')

    op.drop_constraint('fk_creative_assets_approved_attempt_id', 'creative_assets', type_='foreignkey')
    op.drop_index(op.f('ix_creative_assets_concept_id'), table_name='creative_assets')
    op.drop_index(op.f('ix_creative_assets_campaign_id'), table_name='creative_assets')
    op.drop_table('creative_assets')

    op.drop_constraint('fk_campaigns_selected_concept_id', 'campaigns', type_='foreignkey')
    op.drop_index(op.f('ix_creative_concepts_campaign_id'), table_name='creative_concepts')
    op.drop_table('creative_concepts')

    op.drop_index(op.f('ix_campaigns_product_id'), table_name='campaigns')
    op.drop_index(op.f('ix_campaigns_owner_id'), table_name='campaigns')
    op.drop_index(op.f('ix_campaigns_brand_id'), table_name='campaigns')
    op.drop_table('campaigns')

    op.drop_index(op.f('ix_products_brand_id'), table_name='products')
    op.drop_table('products')

    op.drop_index(op.f('ix_brands_owner_id'), table_name='brands')
    op.drop_table('brands')

    op.drop_index(op.f('ix_users_google_sub'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')

    # Drop enum types
    sa.Enum(name='campaign_status').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='concept_status').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='asset_status').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='platform').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='placement').drop(op.get_bind(), checkfirst=True)
