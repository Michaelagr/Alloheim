import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Battery parameters (for display only)
BATTERY_POWER = 100  # kW
BATTERY_CAPACITY = 215  # kWh
DEPTH_OF_DISCHARGE = 90  # %
BATTERY_EFFICIENCY = 0.95

# Page config
st.set_page_config(page_title="Batterie-Analyse für Alloheim Standorte", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_data
def load_summary_results(results_folder):
    """Load pre-calculated summary results"""
    summary_file = Path(results_folder) / "summary_results.csv"
    df = pd.read_csv(summary_file)
    df = df.reset_index(drop=True)  # Ensure clean integer index
    return df


@st.cache_data
def load_location_detail(results_folder, location_id):
    """Load pre-calculated detail results for a specific location"""
    detail_file = Path(results_folder) / f"detail_{location_id}.parquet"
    df = pd.read_parquet(detail_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Handle backwards compatibility with old data format
    # Old format: ps_grid_load, battery_charge, battery_discharge, battery_soc
    # New format: ps_grid_load_pv, ps_grid_load_no_pv, battery_charge_pv, etc.
    if 'ps_grid_load_pv' not in df.columns and 'ps_grid_load' in df.columns:
        # Old format detected - rename to new format
        # Old data only has PV scenario, so we'll use it for both
        df['ps_grid_load_pv'] = df['ps_grid_load']
        df['battery_charge_pv'] = df['battery_charge']
        df['battery_discharge_pv'] = df['battery_discharge']
        df['battery_soc_pv'] = df['battery_soc']

        # For no-PV scenario, use the same data (best we can do with old format)
        # This means "Ohne PV" view will show PV data until analysis is re-run
        df['ps_grid_load_no_pv'] = df['ps_grid_load']
        df['battery_charge_no_pv'] = df['battery_charge']
        df['battery_discharge_no_pv'] = df['battery_discharge']
        df['battery_soc_no_pv'] = df['battery_soc']

    return df


def create_location_chart(df_result, original_peak, optimized_peak, pv_mode):
    """Create interactive Plotly chart for location detail view"""

    fig = go.Figure()

    # Determine which optimized load column to use based on view mode
    if pv_mode == "Mit PV":
        optimized_load_col = 'ps_grid_load_pv'
        chart_title = "Lastprofil mit PV und Batterie"
        optimized_label = 'Netzbezug mit PV + Batterie'
    else:
        optimized_load_col = 'ps_grid_load_no_pv'
        chart_title = "Lastprofil mit Batterie (ohne PV)"
        optimized_label = 'Netzbezug mit Batterie'

    # Original load - #00095B (always show for reference)
    fig.add_trace(go.Scatter(
        x=df_result.index,
        y=df_result['original_load_kw'],
        name='Originallast (ohne Optimierung)',
        line=dict(color='#00095B', width=1.5),
        opacity=0.7
    ))

    # PV if available - #E2EC2B with higher opacity (lighter)
    if 'pv_kw' in df_result.columns and df_result['pv_kw'].sum() > 0 and pv_mode == "Mit PV":
        fig.add_trace(go.Scatter(
            x=df_result.index,
            y=df_result['pv_kw'],
            name='PV-Erzeugung',
            line=dict(color='#E2EC2B', width=1),
            fill='tozeroy',
            opacity=0.3
        ))

        # Net load - #7582F6
        fig.add_trace(go.Scatter(
            x=df_result.index,
            y=df_result['net_load_kw'],
            name='Netzbezug mit PV (ohne Batterie)',
            line=dict(color='#7582F6', width=1.5),
            opacity=0.7
        ))

    # Optimized load - use the correct column based on mode
    fig.add_trace(go.Scatter(
        x=df_result.index,
        y=df_result[optimized_load_col],
        name=optimized_label,
        line=dict(color='#EF553B', width=2.5)
    ))

    # Peak lines - these are already mode-specific from the caller
    fig.add_hline(
        y=original_peak,
        line_dash="dash",
        line_color="gray",
        line_width=2,
        annotation=dict(
            text=f"Spitzenlast Lastprofil ohne Batterie: {original_peak:.1f} kW",
            font=dict(size=13, color="white"),
            bgcolor="gray",
            borderpad=4
        )
    )
    fig.add_hline(
        y=optimized_peak,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation=dict(
            text=f"Spitzenlast mit Batterie (inkl. PV): {optimized_peak:.1f} kW",
            font=dict(size=13, color="white"),
            bgcolor="red",
            borderpad=4
        )
    )

    fig.update_layout(
        title=chart_title,
        xaxis_title="Zeitpunkt",
        yaxis_title="Leistung (kW)",
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        )
    )

    return fig


# Main App
def main():
    st.title("⚡ Alloheim Batteriesimulation Übersicht")
    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        # Page selection
        st.header("Navigation")
        page = st.sidebar.radio("Ansicht", ["📊 Übersicht", "🔍 Standort Details"])
        st.write("---")
        # st.header("Einstellungen")

        # Default to results folder next to the script
        script_dir = Path(__file__).parent
        default_results = script_dir / "results"

        year_option = st.radio(
            "Welches Jahr möchten Sie anzeigen?",
            options=["2024", "2025"],
            horizontal=True,
            index=0
        )
        if year_option == "2024":
            results_folder = str(script_dir / "results")
        else:
            results_folder = str(script_dir / "results_2025")

        if not Path(results_folder).exists():
            st.error("Ergebnis-Ordner nicht gefunden!")
            st.info(f"Erwartet: {Path(results_folder).absolute()}")
            st.info("Bitte führen Sie zuerst `run_analysis.py` aus.")
            return

        st.markdown("---")
        st.markdown("**Batterie Konfiguration**")
        st.metric("Leistung", f"{BATTERY_POWER} kW")
        st.metric("Kapazität", f"{BATTERY_CAPACITY} kWh")
        st.metric("DoD", f"{DEPTH_OF_DISCHARGE}%")
        st.metric("Effizienz", f"{BATTERY_EFFICIENCY * 100}%")

    # Load summary data
    try:
        results_df = load_summary_results(results_folder)
    except Exception as e:
        st.error(f"Fehler beim Laden der Ergebnisse: {str(e)}")
        st.info("Bitte führen Sie zuerst `run_analysis.py` aus.")
        return

    if page == "📊 Übersicht":
        # st.subtitle("Batterie-Simulation")

        # Summary metrics
        col1, col2, col3 = st.columns([2, 3, 3])
        with col1:
            st.metric("Anzahl Standorte", len(results_df))
        with col2:
            st.metric(
                "Mögliche Gesamtersparnis (basierend auf Lastprofilen aus 2024)",
                "€{:,.2f}".format(results_df['savings_no_pv'].sum()).replace(",", "X").replace(".", ",").replace("X",
                                                                                                                 ".")
            )
            durchschnitt_ersparnis = (results_df['savings_no_pv'].mean()).round(2)
            st.metric(
                "Durchschnittliche Ersparnis",
                f"€{durchschnitt_ersparnis:,}".replace(",", "X").replace(".", ",").replace("X", ".")
            )
        # with col3:
        #    st.metric("Durchschnittliche Spitzenlast", f"{results_df['original_peak_no_pv'].mean():,.1f} kW")
        with col3:
            # Battery parameters box
            st.info("""
            **Batterie-Spezifikationen:**
            - Kapazität: 215 kWh
            - Leistung: 100 kW
            - Investitionskosten: ~52.000 € pro Standort
            - Durchschnittliche Installationszeit: 2,5 Monate
            """)

        st.markdown("---")

        # Calculate amortization and annual consumption
        results_df['amortization_years'] = 52000 / results_df['savings_no_pv'].replace(0, float('inf'))

        st.subheader("Standortübersicht (absteigend sortiert nach Ersparnis)")

        # Results table
        display_df = results_df[[
            'location_name', 'demand_charge', 'original_peak_no_pv',
            'savings_no_pv', 'amortization_years'
        ]].copy()

        # Calculate annual consumption - need to load each detail file
        with st.spinner("⏳ Daten werden geladen..."):
            annual_consumption_list = []
            for idx in range(len(results_df)):
                try:
                    location_id = results_df.iloc[idx]['location_id']
                    df_detail = load_location_detail(results_folder, location_id)
                    annual_kwh = df_detail['original_load_kw'].sum() * 0.25
                    annual_consumption_list.append(annual_kwh)
                except:
                    annual_consumption_list.append(0)

        display_df['annual_consumption'] = annual_consumption_list

        # Reorder columns
        display_df = display_df[['location_name', 'demand_charge', 'annual_consumption',
                                 'original_peak_no_pv', 'savings_no_pv', 'amortization_years']]

        display_df.columns = [
            'Standort', 'Leistungspreis (€/kW)', 'Jahresverbrauch (kWh)',
            'Spitzenlast (kW)', 'Ersparnis (€)', 'Amortisationszeit (Jahre)'
        ]

        # Format numbers
        display_df['Leistungspreis (€/kW)'] = display_df['Leistungspreis (€/kW)'].round(2)
        display_df['Jahresverbrauch (kWh)'] = display_df['Jahresverbrauch (kWh)'].round(0).astype(int)
        display_df['Spitzenlast (kW)'] = display_df['Spitzenlast (kW)'].round(1)
        display_df['Ersparnis (€)'] = display_df['Ersparnis (€)'].round(0)
        display_df['Amortisationszeit (Jahre)'] = display_df['Amortisationszeit (Jahre)'].round(1)

        # Sort by savings descending
        display_df = display_df.sort_values('Ersparnis (€)', ascending=False)

        st.dataframe(display_df, height=600)

        # Download button
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tabelle als CSV herunterladen",
            data=csv,
            file_name="peakshaving_results.csv",
            mime="text/csv",
        )

    else:  # Location Details
        st.header("Standort Details")

        # Location selector - cleaner format without first number
        location_options = {}
        for idx in range(len(results_df)):
            # Extract just the location number and name from location_name
            # e.g., "1110 Alloheim Senioren-Residenz "Dormagen""
            location_display = results_df.iloc[idx]['location_name']
            location_options[location_display] = idx

        col11, col21 = st.columns(2)
        with col11:
            selected = st.selectbox("Standort auswählen", list(location_options.keys()))
        # with col21:
        #         view_mode = st.radio(
        #             "Ansicht",
        #             ["Mit PV", "Ohne PV"],
        #             horizontal=True
        #         )

        if selected:
            idx = location_options[selected]
            location_data = results_df.iloc[idx]

            # Load pre-calculated detail data
            try:
                df_detail = load_location_detail(results_folder, location_data['location_id'])
            except Exception as e:
                st.error(f"Fehler beim Laden der Detail-Daten: {str(e)}")
                return

            # Extract clean location name (remove location number prefix)
            clean_name = location_data['location_name']
            # Remove leading numbers like "1110 " if present
            import re
            clean_name = re.sub(r'^\d+\s+', '', clean_name)

            st.write("---")
            # Display location name as subtitle
            st.subheader(clean_name)

            # Display mode selection
            # st.markdown("---")
            # col1, col2 = st.columns([3, 1])

            # with col1:

            # Calculate annual consumption
            annual_consumption = (df_detail['original_load_kw'].sum() * 0.25)

            # Calculate amortization for no-PV scenario
            battery_cost = 52000
            amortization_no_pv = battery_cost / location_data['savings_no_pv'] if location_data[
                                                                                      'savings_no_pv'] > 0 else float(
                'inf')

            # Battery info box
            st.info(f"""
            **Batterie-Spezifikationen:**
            Kapazität: 215 kWh | Leistung: 100 kW | Investitionskosten: {battery_cost:,.0f} € | Amortisationszeit: {amortization_no_pv:.1f} Jahre
            """)

            # Metrics - First row: Without PV scenario
            # st.markdown("---")
            with st.container(border=True):
                st.markdown("**Szenario ohne PV:**")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Jahresverbrauch", f"{annual_consumption:,.0f} kWh")
                with col2:
                    st.metric("Spitzenlast", f"{location_data['original_peak_no_pv']:.1f} kW")
                with col3:
                    st.metric("Spitzenlast mit Batterie", f"{location_data['optimized_peak_no_pv']:.1f} kW",
                              f"{location_data['optimized_peak_no_pv'] - location_data['original_peak_no_pv']:.1f} kW",
                              delta_color="inverse")
                with col4:
                    st.metric("Ersparnis", f"€{location_data['savings_no_pv']:,.0f}")

                st.write("---")
                # Metrics - Second row: PV details and scenario
                st.markdown(f"**Szenario mit PV: {location_data['suggested_kwp']:.1f} kWp):**")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    # Calculate annual consumption with PV as the sum of ps_grid_load_pv [kWh]
                    annual_consumption_pv = df_detail['ps_grid_load_pv'].clip(lower=0).sum() * 0.25
                    st.metric("Jahresverbrauch (Netzbezug) mit PV", f"{annual_consumption_pv:,.0f} kWh")
                with col2:
                    st.metric("Spitzenlast", f"{location_data['original_peak_pv']:.1f} kW")
                with col3:
                    st.metric("Spitzenlast mit Batterie", f"{location_data['optimized_peak_pv']:.1f} kW",
                              f"{location_data['optimized_peak_pv'] - location_data['original_peak_pv']:.1f} kW",
                              delta_color="inverse")
                with col4:
                    st.metric("Ersparnis", f"€{location_data['savings_pv']:,.0f}")

            # Warning if applicable
            # if warning:
            #    st.warning("⚠️ Das volle Potenzial der Batterie für Spitzenreduktion konnte aufgrund der Lastprofilcharakteristika nicht erreicht werden.")
            st.subheader("Lastprofil Analyse")
            view_mode = st.radio(
                "Ansicht",
                ["Mit PV", "Ohne PV"],
                horizontal=True
            )

            # Determine which metrics to use based on view mode
            if view_mode == "Mit PV":
                original_peak = location_data['original_peak_pv']
                optimized_peak = location_data['optimized_peak_pv']
                actual_reduction = location_data['reduction_pv']
                savings = location_data['savings_pv']
                warning = location_data['warning_pv']
                pv_kwp = location_data['suggested_kwp']
            else:
                original_peak = location_data['original_peak_no_pv']
                optimized_peak = location_data['optimized_peak_no_pv']
                actual_reduction = location_data['reduction_no_pv']
                savings = location_data['savings_no_pv']
                warning = location_data['warning_no_pv']
                pv_kwp = 0

            # Chart
            # st.markdown("---")
            fig = create_location_chart(df_detail, original_peak, optimized_peak, view_mode)
            st.plotly_chart(fig)

            # Additional metrics in expandable sections
            with st.expander("📊 Detaillierte Statistiken", expanded=False):
                col1, col2, col3 = st.columns(3)

                # Determine which battery columns to use based on view mode
                if view_mode == "Mit PV":
                    battery_discharge_col = 'battery_discharge_pv'
                    battery_charge_col = 'battery_charge_pv'
                    battery_soc_col = 'battery_soc_pv'
                    optimized_load_col = 'ps_grid_load_pv'
                else:
                    battery_discharge_col = 'battery_discharge_no_pv'
                    battery_charge_col = 'battery_charge_no_pv'
                    battery_soc_col = 'battery_soc_no_pv'
                    optimized_load_col = 'ps_grid_load_no_pv'

                with col1:
                    st.subheader("Batterie")
                    st.write(f"**Max Entladung:** {df_detail[battery_discharge_col].max():.1f} kW")
                    st.write(f"**Max Ladung:** {df_detail[battery_charge_col].max():.1f} kW")
                    st.write(
                        f"**Min SoC:** {df_detail[battery_soc_col].min():.1f} kWh ({df_detail[battery_soc_col].min() / BATTERY_CAPACITY * 100:.1f}%)")
                    st.write(
                        f"**Zyklen (geschätzt):** {(df_detail[battery_discharge_col].sum() * 0.25 / BATTERY_CAPACITY):.1f}")

                with col2:
                    st.subheader("Last")
                    st.write(f"**Original Jahresverbrauch:** {(df_detail['original_load_kw'].sum() * 0.25):,.0f} kWh")
                    st.write(f"**Optimiert Jahresverbrauch:** {(df_detail[optimized_load_col].sum() * 0.25):,.0f} kWh")
                    st.write(f"**Peak Reduktion:** {((original_peak - optimized_peak) / original_peak * 100):.1f}%")

                with col3:
                    if pv_kwp > 0 and 'pv_kw' in df_detail.columns:
                        st.subheader("PV")
                        total_pv_production = df_detail['pv_kw'].sum() * 0.25
                        # Calculate PV export (when PV > load)
                        pv_export = ((df_detail['pv_kw'] - df_detail['original_load_kw']).clip(lower=0).sum() * 0.25)
                        pv_self_consumption = total_pv_production - pv_export

                        st.write(f"**Max PV Erzeugung:** {df_detail['pv_kw'].max():.1f} kW")
                        st.write(f"**Jahres PV Erzeugung:** {total_pv_production:,.0f} kWh")
                        st.write(f"**PV Eigenverbrauch:** {pv_self_consumption:,.0f} kWh")
                        st.write(f"**PV Export:** {pv_export:,.0f} kWh")
                        st.write(f"**Eigenverbrauchsquote:** {(pv_self_consumption / total_pv_production * 100):.1f}%")


if __name__ == "__main__":
    main()