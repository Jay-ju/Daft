use std::sync::Arc;

use common_daft_config::DaftExecutionConfig;
use common_error::DaftResult;
use common_scan_info::ScanState;
use common_treenode::{Transformed, TreeNode};

use super::OptimizerRule;
use crate::{LogicalPlan, SourceInfo};

// Materialize scan tasks from scan operators for all physical scans.
#[derive(Default, Debug)]
pub struct MaterializeScans {
    cfg: Option<Arc<DaftExecutionConfig>>,
}

impl MaterializeScans {
    pub fn new(cfg: Option<Arc<DaftExecutionConfig>>) -> Self {
        Self { cfg }
    }
}

impl OptimizerRule for MaterializeScans {
    fn try_optimize(&self, plan: Arc<LogicalPlan>) -> DaftResult<Transformed<Arc<LogicalPlan>>> {
        plan.transform_up(|node| self.try_optimize_node(node))
    }
}

impl MaterializeScans {
    #[allow(clippy::only_used_in_recursion)]
    fn try_optimize_node(
        &self,
        plan: Arc<LogicalPlan>,
    ) -> DaftResult<Transformed<Arc<LogicalPlan>>> {
        match &*plan {
            LogicalPlan::Source(source) => match &*source.source_info {
                SourceInfo::Physical(_physical_scan_info) => {
                    let source_plan = Arc::unwrap_or_clone(plan);
                    if let LogicalPlan::Source(source) = source_plan {
                        let materialized_source = source.build_materialized_scan_source()?;

                        // If we have config and split pass is available, try to split
                        if let Some(cfg) = &self.cfg
                            && let Some(pass) = common_scan_info::SPLIT_AND_MERGE_PASS.get()
                            && let SourceInfo::Physical(ref info) = *materialized_source.source_info
                            && let ScanState::Tasks(ref tasks) = info.scan_state
                        {
                            let new_scan_tasks = pass(tasks.clone(), &info.pushdowns, cfg)?;

                            let new_info = info.with_scan_state(ScanState::Tasks(new_scan_tasks));
                            let optimized_source = crate::ops::Source::new(
                                materialized_source.output_schema.clone(),
                                Arc::new(SourceInfo::Physical(new_info)),
                            );
                            Ok(Transformed::yes(optimized_source.into()))
                        } else {
                            Ok(Transformed::yes(materialized_source.into()))
                        }
                    } else {
                        unreachable!("This logical plan was already matched as a Source node")
                    }
                }
                _ => Ok(Transformed::no(plan)),
            },
            _ => Ok(Transformed::no(plan)),
        }
    }
}
