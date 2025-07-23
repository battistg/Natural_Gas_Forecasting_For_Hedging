import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class TimingAdvantageHedgingSimulator:
    
    
    def __init__(self, data_path, simulation_years=None, option_maturity_days=7, 
                 exposure_quantity=10000, exposure_direction='LONG'):
        self.data_path = data_path
        self.simulation_years = simulation_years
        self.option_maturity_days = option_maturity_days
        self.exposure_quantity = exposure_quantity
        self.exposure_direction = exposure_direction
        self.results = []
        
    def load_and_prepare_data(self):

        
        # Loading
        df = pd.read_csv(self.data_path)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Calculate returns and volatility using TTF_Spot
        df['Returns'] = np.log(df['TTF_Spot']).diff()
        df['Historical_Volatility'] = df['Returns'].rolling(30).std() * np.sqrt(252)  # 30-day window
        
        # Shift forecasted data to simulate information advantage
        df['Available_Pred_TTF_Spot'] = df['TTF_Spot_Forecasted_2024'].shift(-1)
        df['Available_Pred_Vol_Reg'] = df['Pred_Vol_Reg_2024'].shift(-1)
        df['Available_Pred_Dir_Reg'] = df['Pred_Dir_Reg_2024'].shift(-1)
        
        # Filter for specified years
        df['Year'] = df['Date'].dt.year
        if self.simulation_years is not None:
            if isinstance(self.simulation_years, int):
                self.simulation_years = [self.simulation_years]
            df = df[df['Year'].isin(self.simulation_years)].reset_index(drop=True)
        
        # Clean data
        df = df.dropna(subset=['TTF_Spot', 'Risk_Free_Rate'])
        df = df.reset_index(drop=True)
        
        years_str = str(self.simulation_years) if self.simulation_years else "ALL"
        print(f"Data loaded: {len(df)} observations for years {years_str}")
        print(f"Period: {df['Date'].min().strftime('%Y-%m-%d')} - {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"Exposure direction: {self.exposure_direction}")
        
        self.df = df
        return df
    
    def black_scholes_call(self, S, K, T, r, sigma):
        
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0, 0
        
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            
            return max(call_price, 0), delta
        except:
            return 0, 0
    
    def black_scholes_put(self, S, K, T, r, sigma):
        
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0, 0
        
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
            
            return max(put_price, 0), delta
        except:
            return 0, 0
    
    def get_market_volatility(self, i):
        
        base_vol = self.df.loc[:i]['Returns'].rolling(30).std().iloc[-1] * np.sqrt(365)
        
        if np.isnan(base_vol):
            base_vol = 0.3
            
        return base_vol  
    
    def simple_price_signal(self, i, strategy_type='classic'):
       
        if i < 3:
            return 'NEUTRAL'
    
        if strategy_type == 'classic':
            # CLASSIC
            price_today = self.df.loc[i, 'TTF_Spot']           
            price_yesterday = self.df.loc[i-1, 'TTF_Spot']    
            price_day_before = self.df.loc[i-2, 'TTF_Spot']   
        
            # Calculate returns: today vs yesterday, yesterday vs day before
            return_1 = (price_today - price_yesterday) / price_yesterday
            return_2 = (price_yesterday - price_day_before) / price_day_before
        
            avg_return = (return_1 + return_2) / 2
        
        else:  # forecast
            # FORECAST
            price_today = self.df.loc[i, 'TTF_Spot']                   
            price_yesterday = self.df.loc[i-1, 'TTF_Spot']          
        
            # Forecasted price for tomorrow 
            if i+1 < len(self.df):
                price_tomorrow_forecast = self.df.loc[i, 'Available_Pred_TTF_Spot'] 
            else:
                # If no forecast for tomorrow, use current forecast as proxy
                price_tomorrow_forecast = self.df.loc[i, 'Available_Pred_TTF_Spot']
        
            if pd.isna(price_tomorrow_forecast):
                return 'NEUTRAL'
        
            # Calculate returns: tomorrow_forecast vs today, today vs yesterday
            return_1 = (price_tomorrow_forecast - price_today) / price_today
            return_2 = (price_today - price_yesterday) / price_yesterday
        
            avg_return = (return_1 + return_2) / 2
    
        # Same threshold for both
        if avg_return > 0.015:  
            return 'UPWARD_PRESSURE'
        elif avg_return < -0.015:  
            return 'DOWNWARD_PRESSURE'
        else:
            return 'NEUTRAL'
    
    def simplified_regime_decision(self, vol_regime, dir_regime):
        
        # LONG gas
        
        if vol_regime == 'High':
            if dir_regime in ['Bull', 'Neutral']:
                return 'CALL' 
            else:  # Bear
                return 'NONE'
                
        elif vol_regime == 'Medium':
            if dir_regime == 'Bull':
                return 'CALL' 
            else:  # Neutral or Bear
                return 'BORDERLINE'  
                
        else:  # Low vol
            return 'NONE'  
        
        return 'NONE'
    
    def integrated_decision(self, regime_decision, price_signal):
        
        if regime_decision in ['CALL', 'PUT']:
   
            return regime_decision.lower(), 'regime_primary'
            
        elif regime_decision == 'BORDERLINE':
            # Use price signal for border cases
            if price_signal == 'UPWARD_PRESSURE':
                return 'call', 'price_tiebreaker'  
            else:
                return 'none', 'price_tiebreaker'  
                    
        else:  # regime_decision == 'NONE'
            
            return 'none', 'regime_primary'
    
    def calculate_simplified_strikes(self, current_price, vol_regime):
        
        if vol_regime == 'High':
            # High volatility
            call_strike = current_price * 1.02  
            put_strike = current_price * 0.98   
        elif vol_regime == 'Medium':
            # Medium volatility
            call_strike = current_price * 1.025 
            put_strike = current_price * 0.975  
        else:  # Low
            # Low volatility
            call_strike = current_price * 1.03  
            put_strike = current_price * 0.97   
            
        return call_strike, put_strike
    
    def get_decision_regimes(self, i):
        
        # Observed regimes (t)
        observed_vol = self.df.loc[i, 'Obs_Vol_Reg'] if not pd.isna(self.df.loc[i, 'Obs_Vol_Reg']) else 'Medium'
        observed_dir = self.df.loc[i, 'Obs_Dir_Reg'] if not pd.isna(self.df.loc[i, 'Obs_Dir_Reg']) else 'Neutral'
        
        # Forecasted regimes 
        forecasted_vol = self.df.loc[i, 'Available_Pred_Vol_Reg'] if not pd.isna(self.df.loc[i, 'Available_Pred_Vol_Reg']) else 'Medium'
        forecasted_dir = self.df.loc[i, 'Available_Pred_Dir_Reg'] if not pd.isna(self.df.loc[i, 'Available_Pred_Dir_Reg']) else 'Neutral'
        
        # CLASSIC: always observed regimes
        classic_vol = observed_vol
        classic_dir = observed_dir
        
        # FORECAST
        regimes_equal = (forecasted_vol == observed_vol) and (forecasted_dir == observed_dir)
        
        if regimes_equal:
            # No change predicted
            forecast_vol = observed_vol
            forecast_dir = observed_dir
            forecast_advantage = False
            regime_change_predicted = False
        else:
            # Regime change predicted
            forecast_vol = forecasted_vol
            forecast_dir = forecasted_dir
            forecast_advantage = True
            regime_change_predicted = True
        
        return {
            'classic': {'vol': classic_vol, 'dir': classic_dir},
            'forecast': {'vol': forecast_vol, 'dir': forecast_dir},
            'forecast_advantage': forecast_advantage,
            'regime_change_predicted': regime_change_predicted,
            'regimes_equal': regimes_equal
        }
    
    def simulate_strategy(self, strategy_type='classic'):
       
        results = []
        current_positions = {}
        cash_account = 0
        cumulative_pnl = 0
        total_protection_cost = 0
        forecast_advantages_used = 0
        regime_changes_predicted = 0
        price_advantages_used = 0
        
        decision_days_count = 0
        
        for i in range(3, len(self.df)): 
            today_data = self.df.loc[i]
            yesterday_data = self.df.loc[i-1]
            
            current_spot = today_data['TTF_Spot']
            current_date = today_data['Date']
            risk_free_rate = yesterday_data['Risk_Free_Rate'] / 100
            
            daily_pnl = 0
            option_pnl = 0
            exposure_pnl = 0
            
            
            # EXPOSURE P&L CALCULATION
            price_change = current_spot - yesterday_data['TTF_Spot']
            
            
            exposure_pnl = -price_change * self.exposure_quantity
            

            # EXISTING POSITIONS MANAGEMENT
            positions_to_close = []
            
            for option_id, position in current_positions.items():
                position['days_to_maturity'] -= 1
                
                T = max(position['days_to_maturity'] / 365, 0.001)
                

                vol_estimate = self.get_market_volatility(i)
                
                if position['option_type'] == 'call':
                    new_option_price, _ = self.black_scholes_call(
                        S=current_spot, K=position['strike'], T=T, 
                        r=risk_free_rate, sigma=vol_estimate
                    )
                else:  # put
                    new_option_price, _ = self.black_scholes_put(
                        S=current_spot, K=position['strike'], T=T, 
                        r=risk_free_rate, sigma=vol_estimate
                    )
                
                option_value_change = new_option_price - position['last_option_price']
                position_option_pnl = option_value_change * position['quantity']
                
                position['last_option_price'] = new_option_price
                position['cumulative_pnl'] += position_option_pnl
                
                option_pnl += position_option_pnl
                
                # Check expiration
                if position['days_to_maturity'] <= 0:
                    if position['option_type'] == 'call':
                        intrinsic_value = max(current_spot - position['strike'], 0)
                    else:  # put
                        intrinsic_value = max(position['strike'] - current_spot, 0)
                    
                    settlement_pnl = intrinsic_value * position['quantity']
                    position['final_pnl'] = position['cumulative_pnl'] + settlement_pnl
                    option_pnl += settlement_pnl
                    
                    positions_to_close.append(option_id)
            
            for option_id in positions_to_close:
                del current_positions[option_id]
            

            # DAILY DECISIONS 
            
            forecast_advantage_used = False
            regime_change_predicted = False
            price_advantage_used = False
            
           
            is_decision_day = len(current_positions) == 0
            
            if is_decision_day:
                decision_days_count += 1
                
                # Get regimes for decision
                regimes_info = self.get_decision_regimes(i)
                
                if strategy_type == 'classic':
                    decision_vol = regimes_info['classic']['vol']
                    decision_dir = regimes_info['classic']['dir']
                else:  # forecast
                    decision_vol = regimes_info['forecast']['vol']
                    decision_dir = regimes_info['forecast']['dir']
                    forecast_advantage_used = regimes_info['forecast_advantage']
                    regime_change_predicted = regimes_info['regime_change_predicted']
                
                # Simple price signal
                price_signal = self.simple_price_signal(i, strategy_type)
                
                # Track if forecast uses price advantage
                if strategy_type == 'forecast' and price_signal != self.simple_price_signal(i, 'classic'):
                    price_advantage_used = True
                
                # PRIMARY regime decision
                regime_decision = self.simplified_regime_decision(decision_vol, decision_dir)
                
                # INTEGRATED decision (regime + price hierarchy)
                final_protection_type, signal_source = self.integrated_decision(regime_decision, price_signal)
                
                if final_protection_type != 'none':
                    # BINARY protection
                    protection_quantity = self.exposure_quantity
                    
                    call_strike, put_strike = self.calculate_simplified_strikes(current_spot, decision_vol)
                    
                
                    vol_for_pricing = self.get_market_volatility(i)
                    T = self.option_maturity_days / 365
                    
                    if final_protection_type == 'call':
                        call_price, _ = self.black_scholes_call(
                            S=current_spot, K=call_strike, T=T, 
                            r=risk_free_rate, sigma=vol_for_pricing  
                        )
                        
                        if call_price > 0:
                            actual_cost = call_price * protection_quantity
                            cash_account -= actual_cost
                            total_protection_cost += actual_cost
                            
                            option_id = f"call_{current_date.strftime('%Y%m%d')}"
                            current_positions[option_id] = {
                                'option_type': 'call',
                                'strike': call_strike,
                                'quantity': protection_quantity,
                                'premium_paid': call_price,
                                'days_to_maturity': self.option_maturity_days,
                                'last_option_price': call_price,
                                'cumulative_pnl': -actual_cost,
                                'start_date': current_date,
                                'vol_regime_used': decision_vol,  
                                'forecast_advantage': forecast_advantage_used,
                                'price_advantage': price_advantage_used,
                                'signal_source': signal_source
                            }
                    
                    elif final_protection_type == 'put':
                        put_price, _ = self.black_scholes_put(
                            S=current_spot, K=put_strike, T=T, 
                            r=risk_free_rate, sigma=vol_for_pricing 
                        )
                        
                        if put_price > 0:
                            actual_cost = put_price * protection_quantity
                            cash_account -= actual_cost
                            total_protection_cost += actual_cost
                            
                            option_id = f"put_{current_date.strftime('%Y%m%d')}"
                            current_positions[option_id] = {
                                'option_type': 'put',
                                'strike': put_strike,
                                'quantity': protection_quantity,
                                'premium_paid': put_price,
                                'days_to_maturity': self.option_maturity_days,
                                'last_option_price': put_price,
                                'cumulative_pnl': -actual_cost,
                                'start_date': current_date,
                                'vol_regime_used': decision_vol,
                                'forecast_advantage': forecast_advantage_used,
                                'price_advantage': price_advantage_used,
                                'signal_source': signal_source
                            }
            
            # Update counters
            if forecast_advantage_used:
                forecast_advantages_used += 1
            if regime_change_predicted:
                regime_changes_predicted += 1
            if price_advantage_used:
                price_advantages_used += 1
            

            # TOTAL P&L CALCULATIOn
            
            daily_pnl = exposure_pnl + option_pnl
            cumulative_pnl += daily_pnl
            
            interest_income = cash_account * (risk_free_rate / 365)
            cash_account += interest_income
            
            total_positions = len(current_positions)
            total_protection_value = sum([pos['last_option_price'] * pos['quantity'] 
                                        for pos in current_positions.values()])
            
            results.append({
                'Date': current_date,
                'Strategy': strategy_type,
                'Spot_Price': current_spot,
                'Daily_PnL': daily_pnl,
                'Exposure_PnL': exposure_pnl,
                'Option_PnL': option_pnl,
                'Cumulative_PnL': cumulative_pnl,
                'Cash_Account': cash_account,
                'Interest_Income': interest_income,
                'Active_Positions': total_positions,
                'Total_Protection_Value': total_protection_value,
                'Total_Protection_Cost': total_protection_cost,
                'Is_Decision_Day': is_decision_day,
                'Forecast_Advantage_Used': forecast_advantage_used,
                'Regime_Change_Predicted': regime_change_predicted,
                'Price_Advantage_Used': price_advantage_used,
                'Signal_Source': signal_source if is_decision_day and 'signal_source' in locals() else 'no_signal',
                'Price_Signal': price_signal if is_decision_day and 'price_signal' in locals() else 'neutral',
                'Protection_Type': final_protection_type if is_decision_day and 'final_protection_type' in locals() else 'none'
            })
        
        return pd.DataFrame(results)
    
    def simulate_no_hedging(self):
        results = []
        cumulative_pnl = 0
        
        for i in range(3, len(self.df)): 
            today_data = self.df.loc[i]
            yesterday_data = self.df.loc[i-1]
            
            current_spot = today_data['TTF_Spot']
            current_date = today_data['Date']
            
            price_change = current_spot - yesterday_data['TTF_Spot']
            

            exposure_pnl = -price_change * self.exposure_quantity
            
            daily_pnl = exposure_pnl
            cumulative_pnl += daily_pnl
            
            results.append({
                'Date': current_date,
                'Strategy': 'no_hedging',
                'Spot_Price': current_spot,
                'Daily_PnL': daily_pnl,
                'Exposure_PnL': exposure_pnl,
                'Option_PnL': 0,
                'Cumulative_PnL': cumulative_pnl,
                'Cash_Account': 0,
                'Interest_Income': 0,
                'Active_Positions': 0,
                'Total_Protection_Value': 0,
                'Total_Protection_Cost': 0,
                'Is_Decision_Day': False,
                'Forecast_Advantage_Used': False,
                'Regime_Change_Predicted': False,
                'Price_Advantage_Used': False,
                'Signal_Source': 'no_hedging',
                'Price_Signal': 'neutral',
                'Protection_Type': 'none'
            })
        
        return pd.DataFrame(results)
    
    def run_simulation(self):
        print(f"Exposure: {self.exposure_direction} {self.exposure_quantity:,} MWh")
        print(f"Option Duration: {self.option_maturity_days} days")
        
        self.load_and_prepare_data()
        
        print("\nSimulating classic strategy...")
        classic_results = self.simulate_strategy('classic')
        
        print("Simulating forecast strategy...")
        forecast_results = self.simulate_strategy('forecast')
        
        print("Simulating no hedging...")
        no_hedging_results = self.simulate_no_hedging()
        
        self.classic_results = classic_results
        self.forecast_results = forecast_results
        self.no_hedging_results = no_hedging_results
        
        self.analyze_results()
        
        return classic_results, forecast_results, no_hedging_results
    
    def analyze_results(self):
       
        
        c_results = self.classic_results
        f_results = self.forecast_results
        n_results = self.no_hedging_results
        
        # Calculate performance metrics for each strategy
        strategies = {
            'No Hedging': n_results,
            'Classic Protection': c_results,
            'Forecast Protection': f_results
        }
        
        print(f"\nPERFORMANCE RESULTS")
        print("="*50)
        
        # P&L
        classic_final = c_results['Cumulative_PnL'].iloc[-1]
        forecast_final = f_results['Cumulative_PnL'].iloc[-1]
        no_hedge_final = n_results['Cumulative_PnL'].iloc[-1]
        
        print(f"FINAL P&L:")
        print(f"No Hedging:           {no_hedge_final:+,.2f}")
        print(f"Classic Protection:   {classic_final:+,.2f}")
        print(f"Forecast Protection:  {forecast_final:+,.2f}")
        
        # Maximum Drawdown
        def calculate_max_drawdown(results):
            cumulative = results['Cumulative_PnL']
            running_max = cumulative.cummax()
            drawdown_absolute = cumulative - running_max
            max_drawdown_absolute = drawdown_absolute.min()
            return max_drawdown_absolute
        
        classic_dd = calculate_max_drawdown(c_results)
        forecast_dd = calculate_max_drawdown(f_results)
        no_hedge_dd = calculate_max_drawdown(n_results)
        
        print(f"\nMAXIMUM DRAWDOWN:")
        print(f"No Hedging:           {no_hedge_dd:+,.2f}")
        print(f"Classic Protection:   {classic_dd:+,.2f}")
        print(f"Forecast Protection:  {forecast_dd:+,.2f}")
        
        # Sharpe Ratio
        def calculate_sharpe_ratio(results):
            daily_returns = results['Daily_PnL']
            mean_return = daily_returns.mean()
            std_return = daily_returns.std()
            
            try:
                avg_risk_free_rate = self.df['Risk_Free_Rate'].mean() / 100 / 365
            except:
                avg_risk_free_rate = 0.02 / 365
                
            sharpe_ratio = (mean_return - avg_risk_free_rate) / std_return if std_return != 0 else 0
            return sharpe_ratio
        
        classic_sharpe = calculate_sharpe_ratio(c_results)
        forecast_sharpe = calculate_sharpe_ratio(f_results)
        no_hedge_sharpe = calculate_sharpe_ratio(n_results)
        
        print(f"\nSHARPE RATIO:")
        print(f"No Hedging:           {no_hedge_sharpe:.4f}")
        print(f"Classic Protection:   {classic_sharpe:.4f}")
        print(f"Forecast Protection:  {forecast_sharpe:.4f}")
        
        # Protection Effectiveness
        def calculate_protection_effectiveness(hedged_results, unhedged_results):
            unhedged_losses = unhedged_results['Daily_PnL'][unhedged_results['Daily_PnL'] < 0].sum()
            hedged_losses = hedged_results['Daily_PnL'][hedged_results['Daily_PnL'] < 0].sum()
            
            if unhedged_losses != 0:
                protection_effectiveness = (hedged_losses / unhedged_losses)
            else:
                protection_effectiveness = 0
                
            return protection_effectiveness
        
        classic_prot_eff = calculate_protection_effectiveness(c_results, n_results)
        forecast_prot_eff = calculate_protection_effectiveness(f_results, n_results)
        
        print(f"\nPROTECTION EFFECTIVENESS:")
        print(f"Classic Protection:   {classic_prot_eff:.4f}")
        print(f"Forecast Protection:  {forecast_prot_eff:.4f}")
        
        self.plot_results()
    
    def plot_results(self):

    
        # 1. Cumulative P&L
        plt.figure(figsize=(9, 6))
        plt.plot(self.classic_results['Date'], self.classic_results['Cumulative_PnL'], 
                 label='Classic Protection', linewidth=2, color='blue')
        plt.plot(self.forecast_results['Date'], self.forecast_results['Cumulative_PnL'], 
                 label='Forecast Protection', linewidth=2, color='red')
        plt.plot(self.no_hedging_results['Date'], self.no_hedging_results['Cumulative_PnL'], 
                 label='No Hedging', linewidth=2, color='green', linestyle='--')
    
        plt.title(f'Cumulative P&L Comparison')
        plt.ylabel('Cumulative P&L')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
        # 2. Cumulative Protection Costs
        plt.figure(figsize=(9, 6))
        plt.plot(self.classic_results['Date'], self.classic_results['Total_Protection_Cost'], 
                 label='Classic Costs', linewidth=2, color='blue')
        plt.plot(self.forecast_results['Date'], self.forecast_results['Total_Protection_Cost'], 
                 label='Forecast Costs', linewidth=2, color='red')
        plt.title('Cumulative Protection Costs\n(Market-Based Pricing)')
        plt.ylabel('Total Cost')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Execution

if __name__ == "__main__":
    DATA_PATH = 'data.csv'
    
    # Simul. year
    SIMULATION_YEARS = [2024]
    
    # Parameters
    OPTION_MATURITY = 30
    EXPOSURE_QUANTITY = 50    # MWh 
    EXPOSURE_DIRECTION = 'LONG'  
    
    simulator = TimingAdvantageHedgingSimulator(
        data_path=DATA_PATH,
        simulation_years=SIMULATION_YEARS,
        option_maturity_days=OPTION_MATURITY,
        exposure_quantity=EXPOSURE_QUANTITY,
        exposure_direction=EXPOSURE_DIRECTION
    )
    
    # Run
    classic_results, forecast_results, no_hedging_results = simulator.run_simulation()
    
